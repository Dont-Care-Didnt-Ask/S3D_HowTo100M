import os
import json
import argparse
from typing import List

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

import modeling.util as util
from modeling.s3dg import S3D
from modeling.prompts import FRANKA_PROMPT_SET

def parse_args():
    # These help messages may be misleading, as they are copied from another script
    parser = argparse.ArgumentParser(description="Compare trajectory with given prompts")

    parser.add_argument("-t", "--trajectories-path", help="Path to a txt file, contaning paths to directories, containing trajectories in avi format.", required=True)
    parser.add_argument("-p", "--prompt-set", default="franka", help="Prompt set to use in evaluation. Defined in modeling/prompts.py")
    parser.add_argument("-e", "--experiment-id", help="Name of current experiment (used to save the results)", required=True)
    parser.add_argument("-o", "--output-dir", help="Directory to save evaluation results.", default="evaluation_results")
    parser.add_argument("--average-by-video", action="store_true")
    parser.add_argument("--average-by-prompt", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--n-frames", type=int, default=32)
    parser.add_argument("--max-n-videos", type=int, default=10, help="Max number of videos taken from each directory when `--average-by-video` is True")
    parser.add_argument("--model-checkpoint-path", default="checkpoints/s3d_howto100m.pth")

    args = parser.parse_args()
    return args

def uniformly_sample_n_frames(video, n_frames):
    # Probably not most accurate frame sampling -- might be improved
    length = video.shape[0]
    step_size = length // n_frames
    return video[::step_size][:n_frames]

def prepare_videos(videos: List[np.ndarray], verbose: bool) -> torch.Tensor:
    videos = torch.from_numpy(np.stack(videos))
    batch_size, n_frames, height, width, n_channels = videos.shape

    if verbose:
        print("Initial videos shape:", videos.shape, " dtype:", videos.dtype)

    if videos.dtype not in (torch.float16, torch.float32, torch.float64):
        videos = videos.float() / 255
    
    video = F.interpolate(
        videos.flatten(0,1).permute(0, 3, 1, 2), # [batch_size * n_frames, n_channels, height, width]
        mode="bicubic",
        size=(224, 224),
    )
    
    videos = videos.reshape(batch_size, n_frames, n_channels, height, width).transpose(1,2)

    if verbose:
        print("Min and max before clipping:", videos.min(), videos.max())

    videos = videos.clamp(0, 1)
    
    if verbose:
        print("Final videos shape:", videos.shape, " dtype:", videos.dtype)

    return videos

def load_model(model_checkpoint_path):
    # Instantiate the model
    embedding_dim = 512
    net = S3D('checkpoints/s3d_dict.npy', embedding_dim)
    # Load the model weights
    net.load_state_dict(torch.load(model_checkpoint_path))
    # Evaluation mode
    net = net.eval()
    return net

def aggregate_similarities(similarities: np.ndarray, group_borders: List[int]):
    """
    ! doc string may be inaccurate, function behavior can change
    Aggregates similarities over slight variations in videos and prompts.
    Returns average similatity of the whole video batch to each prompt group, and its standard error.
    """
    n_videos, n_total_prompts = similarities.shape

    average_similarities = np.empty(len(group_borders) - 1)
    std_similarities = np.empty(len(group_borders) - 1)

    for i in range(len(group_borders) - 1):
        start, end = group_borders[i], group_borders[i + 1]
        #average_similarities[i] = similarities[:, start:end].mean()
        average_similarities[i] = similarities[:, start].mean()
        std_similarities[i] = similarities[:, start].std()

    return average_similarities, std_similarities

def aggregate_similarities_many_video_groups(similarities: np.ndarray, prompt_group_borders: List[int], video_group_borders: List[int]):
    """
    ! doc string may be inaccurate, function behavior can change
    Aggregates similarities over slight variations in videos and prompts.
    Returns average similatity of the whole video batch to each prompt group, and its standard error.
    """
    n_total_videos, n_total_prompts = similarities.shape
    n_video_groups = len(video_group_borders) - 1
    n_prompt_groups = len(prompt_group_borders) - 1

    average_similarities = np.empty((n_video_groups, n_prompt_groups))
    std_similarities = np.empty((n_video_groups, n_prompt_groups))

    for video_group_idx in range(n_video_groups):
        for prompt_group_idx in range(n_prompt_groups):
            p_start, p_end = prompt_group_borders[prompt_group_idx], prompt_group_borders[prompt_group_idx + 1]
            v_start, v_end = video_group_borders[video_group_idx], video_group_borders[video_group_idx + 1]

            average_similarities[video_group_idx, prompt_group_idx] = similarities[v_start:v_end, p_start:p_end].mean()
            std_similarities[video_group_idx, prompt_group_idx] = similarities[v_start:v_end, p_start:p_end].std()

    return average_similarities, std_similarities


@torch.inference_mode()
def main():
    args = parse_args()
    if args.verbose:
        print(f"Running S3D evaluator with following args:\n{args}")

    # Prepare videos
    with open(args.trajectories_path, "r") as f:
        video_dir_paths = [filepath.rstrip("\n") for filepath in f.readlines()]

    video_paths = []
    video_group_borders = [0]
    
    for dir_path in video_dir_paths:
        # Sorted to ensure deterministic results
        video_paths_group = sorted([f"{dir_path}/{path}" for path in os.listdir(dir_path, ) if path.endswith(".avi")])
        video_paths.extend(video_paths_group[:args.max_n_videos] if args.average_by_video else video_paths_group[:1])
        video_group_borders.append(len(video_paths))

    videos = [uniformly_sample_n_frames(util.load_video(path), args.n_frames) for path in video_paths]

    # Video input should be of size Batch x 3 x T x H x W and normalized to [0, 1] 
    # Also, afaik expects either 32 or 16 frames
    videos = prepare_videos(videos, args.verbose)
    
    # Prepare prompts
    if args.prompt_set == "franka":
        prompt_group_names = list(FRANKA_PROMPT_SET.keys())
        prompt_groups = list(FRANKA_PROMPT_SET.values())
    else:
        raise ValueError(f"Prompt set {args.prompt_set} is not supported yet.")
    
    if not args.average_by_prompt:
        prompt_groups = [group[:1] for group in prompt_groups]

    flattened_prompts = [prompt for group in prompt_groups for prompt in group]
    prompt_group_borders = np.cumsum([0] + [len(group) for group in prompt_groups])

    # Prepare model
    net = load_model(args.model_checkpoint_path)

    # Video inference
    if args.verbose:
        print("Embedding videos...")
    video_output = net(videos)
    v_embed = video_output["video_embedding"] / video_output["video_embedding"].norm(p=2, dim=-1, keepdim=True)
    
    # Text inference
    if args.verbose:
        print("Embedding text...")
    text_output = net.text_module(flattened_prompts)
    p_embeds = text_output["text_embedding"] / text_output["text_embedding"].norm(p=2, dim=-1, keepdim=True)

    similarities = (v_embed @ p_embeds.T)
    if args.verbose:
        print("similarities.shape:", similarities.shape)

    # Save artifacts
    result_dir = f"{args.output_dir}/{args.experiment_id}"
    os.makedirs(result_dir, exist_ok=True)
    np.save(f"{result_dir}/similarities.npy", similarities.cpu().numpy())

    context = {
        "prompt_groups": prompt_groups,
        "video_groups": [dir_path.split("/")[-1] for dir_path in video_dir_paths],
    } 
    context = context | vars(args)

    with open(f"{result_dir}/context.json", "w") as f:
        json.dump(context, f, indent=2)

    # visualization
    average_similarities, std_similarities = aggregate_similarities_many_video_groups(
        similarities, 
        prompt_group_borders, 
        video_group_borders, 
    )

    n_cols = 2
    n_rows = (len(video_dir_paths) + n_cols - 1) // n_cols
    plt.figure(figsize=(12, 3 + 3 * n_rows))
    plt.axis("off")

    for i, dir_path in enumerate(video_dir_paths):
        indices = range(len(prompt_groups))
        plt.subplot(n_rows, n_cols, i + 1)
        plt.title(dir_path.split('/')[-1])
        plt.bar(indices, average_similarities[i])
        plt.errorbar(indices, average_similarities[i], yerr=std_similarities[i], fmt="o", color="r")
        plt.xticks(indices, labels=prompt_group_names, rotation=30, ha="right")
    
    plt.suptitle(args.experiment_id)
    plt.tight_layout()
    plt.savefig(f"{result_dir}/aggregated_similarities.png", dpi=350)

if __name__ == "__main__":
    main()
