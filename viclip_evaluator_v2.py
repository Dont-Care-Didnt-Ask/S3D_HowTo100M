import os
import json
import argparse
from typing import List

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

import modeling.util as util
from modeling.viclip import get_viclip
from modeling.prompts import FRANKA_PROMPT_SET, FRANKA_BASELINE

def parse_args():
    parser = argparse.ArgumentParser(description="Compare groups of trajectories from given directories with a prompt set.")

    parser.add_argument("-t", "--trajectories-path", help="Path to a txt file, contaning paths to directories, containing trajectories in avi format.", required=True)
    parser.add_argument("-p", "--prompt-set", default="franka", help="Prompt set to use in evaluation. Defined in modeling/prompts.py")
    parser.add_argument("-e", "--experiment-id", help="Name of current experiment (used to save the results)", required=True)
    parser.add_argument("-o", "--output-dir", help="Directory to save evaluation results.", default="evaluation_results")
    parser.add_argument("--normalize-similarities", action="store_true", help="Whether to normalize similarity scores to account for base rate bias. Advised to be used with --average-by-video.")
    parser.add_argument("--average-by-video", action="store_true")
    parser.add_argument("--average-by-prompt", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.0, help="Coefficient for projection similarity. 0 means no projection, 1 means projection-only.")
    parser.add_argument("--similarity-type", default="cosine", help="Which type of similarity to use.")
    parser.add_argument("--n-frames", type=int, default=8)
    parser.add_argument("--max-n-videos", type=int, default=10, help="Max number of videos taken from each directory when `--average-by-video` is True")
    parser.add_argument("--model-checkpoint-path", default="checkpoints/ViClip-InternVid-10M-FLT.pth")

    args = parser.parse_args()
    return args

def prepare_videos(videos: List[np.ndarray], verbose: bool) -> torch.Tensor:
    videos = torch.from_numpy(np.stack(videos))
    batch_size, n_frames, height, width, n_channels = videos.shape

    if verbose:
        print("Initial videos shape:", videos.shape, " dtype:", videos.dtype)

    if videos.dtype not in (torch.float16, torch.float32, torch.float64):
        v_mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 1, 1, 1, 3)
        v_std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 1, 1, 1, 3)
        videos = (videos.float() / 255 - v_mean) / v_std
    
    videos = F.interpolate(
        videos.flatten(0,1).permute(0, 3, 1, 2), # [batch_size * n_frames, n_channels, height, width]
        mode="bicubic",
        size=(224, 224),
    )
    
    videos = videos.reshape(batch_size, n_frames, n_channels, 224, 224)

    if verbose:
        print("Min and max before clipping:", videos.min(), videos.max())

    videos = videos.clamp(0, 1)
    
    if verbose:
        print("Final videos shape:", videos.shape, " dtype:", videos.dtype)

    return videos

def load_model_and_tokenizer(path: str):
    model = get_viclip(pretrain=path)
    return model["viclip"], model["tokenizer"]

@torch.inference_mode()
def main():
    args = parse_args()
    if args.verbose:
        print(f"Running ViCLIP evaluator with following args:\n{args}")

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

    videos = [util.uniformly_sample_n_frames(util.load_video(path), args.n_frames) for path in video_paths]

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
    viclip, _ = load_model_and_tokenizer(args.model_checkpoint_path)

    # Video inference
    with torch.no_grad():
        if torch.cuda.is_available():
            device = "cuda:0"
            viclip.to(device)
            videos = videos.to(device)

        v_embeds = F.normalize(viclip.encode_vision(videos, test=True), dim=-1)
        p_embeds = F.normalize(viclip.encode_text(flattened_prompts), dim=-1)
        baseline_embed = F.normalize(viclip.encode_text(FRANKA_BASELINE), dim=-1)
        directions = F.normalize(p_embeds - baseline_embed, dim=-1)

        if args.similarity_type == "cosine":
            similarities = util.compute_projection_similarity(p_embeds, v_embeds, directions, args.alpha)
        elif args.similarity_type == "original":
            similarities = util.compute_orig_reward(p_embeds, v_embeds, directions, args.alpha)
        else:
            raise ValueError(f"similarity_type {args.similarity_type} is not supported yet.")

    if args.verbose:
        print("similarities.shape:", similarities.shape)

    # Save artifacts
    result_dir = f"{args.output_dir}/{args.experiment_id}"
    os.makedirs(result_dir, exist_ok=True)
    np.save(f"{result_dir}/similarities.npy", similarities.cpu().numpy())

    sample_video = (videos[0].permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
    util.save_video(sample_video, f"{result_dir}/sample_preprocessed_video.mp4", fps=8)

    context = {
        "prompt_groups": prompt_groups,
        "video_groups": [dir_path.split("/")[-1] for dir_path in video_dir_paths],
    } 
    context = context | vars(args)

    with open(f"{result_dir}/context.json", "w") as f:
        json.dump(context, f, indent=2)

    # visualization
    average_similarities, std_similarities = util.aggregate_similarities_many_video_groups(
        similarities, 
        prompt_group_borders,
        video_group_borders,
        do_normalize=args.normalize_similarities,
    )

    video_group_names = [path.split("/")[-1] for path in video_dir_paths]
    if args.prompt_set == "franka":
        # "friday_t-kettle_d-microwave" -> "kettle_d"
        video_group_names = [name.split("-")[1] for name in video_group_names]

    util.make_barplots(average_similarities, std_similarities, video_group_names, prompt_group_names, args.experiment_id, result_dir)

if __name__ == "__main__":
    main()
