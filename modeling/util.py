import os
import imageio.v3 as iio
from typing import List, Callable, Tuple, Union

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_video(path: str) -> np.ndarray:
    if path.endswith(".mp4"):
        return iio.imread(path, plugin="pyav")
    elif path.endswith(".avi"):
        return iio.imread(path)

def save_video(video, path: str, fps: int):
    iio.imwrite(path, video, fps=fps)

def load_prompts(path: str, verbose: bool) -> List[str]:
    prompts = []

    with open(path, "r") as f:
        for line in f.readlines():
            prompts.append(line.rstrip("\n"))

    if verbose:
        print("Loaded promts:")
        for i, p in enumerate(prompts):
            print(f"{i:2d}: {p}")

    return prompts

def get_video_batch(trajectories_path: str, prepare_video: Callable, n_frames: int, verbose: bool = False) -> torch.Tensor:
    """ Reads a list of video paths, loads videos, preprocess them with `prepare_video` function and arranges them in a batch."""
    with open(trajectories_path, "r") as f:
        video_paths = [line.rstrip("\n") for line in f.readlines()]

    # Preprocessing can be more efficient if done for the whole batch simultaniously
    videos = torch.cat([
        prepare_video(load_video(p), n_frames=n_frames, verbose=verbose) for p in video_paths
    ], dim=0)

    return videos, video_paths

def uniformly_sample_n_frames(video: Union[torch.Tensor, np.ndarray], n_frames: int) -> Union[torch.Tensor, np.ndarray]:
    # Probably not most accurate frame sampling -- might be improved
    length = video.shape[0]
    step_size = length // n_frames
    return video[::step_size][:n_frames]

def make_heatmap(similarity_matrix: np.ndarray, trajectories_names: List[str], labels: List[str], result_dir: str, experiment_id: str):
    sns.heatmap(similarity_matrix, annot=True, fmt=".3f", cmap="crest", xticklabels=labels, yticklabels=trajectories_names)
    plt.title(experiment_id)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    plt.savefig(f"{result_dir}/{experiment_id}.png", dpi=350)

def make_barplots(average_similarities, std_similarities, video_group_names, prompt_group_names, experiment_id, result_dir):
    n_cols = 2
    n_rows = (len(video_group_names) + n_cols - 1) // n_cols
    plt.figure(figsize=(12, 3 + 3 * n_rows))
    plt.axis("off")

    for i, dir_path in enumerate(video_group_names):
        indices = range(len(prompt_group_names))
        plt.subplot(n_rows, n_cols, i + 1)
        plt.title(dir_path.split('/')[-1])
        plt.bar(indices, average_similarities[i])
        plt.errorbar(indices, average_similarities[i], yerr=std_similarities[i], fmt="o", color="r")
        plt.xticks(indices, labels=prompt_group_names, rotation=30, ha="right")
    
    plt.suptitle(experiment_id)
    plt.tight_layout()
    plt.savefig(f"{result_dir}/aggregated_similarities.png", dpi=350)

def aggregate_similarities(similarities: np.ndarray, group_borders: List[int]) -> Tuple[np.ndarray, np.ndarray]:
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

def aggregate_similarities_many_video_groups(similarities: np.ndarray, prompt_group_borders: List[int], video_group_borders: List[int], do_normalize: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes a all-to-all similarity matrix between grouped videos and prompts. Aggregates the similarities by groups.
    Input:
        similarities: (n_total_videos, n_total_prompts)
        prompt_group_borders: List[int] of length (n_prompt_groups + 1) -- prompt group `i` is between `prompt_group_borders[i]` and `prompt_group_borders[i+1]`
        video_group_borders: List[int] of length (n_video_groups + 1) -- analogous
        do_normalize: bool -- whether to normalize similarity for each prompt over all videos

    Output:
        average_similarities: (n_video_groups, n_prompt_groups) -- average similarity between a group of videos and prompts
        std_similarities: (n_video_groups, n_prompt_groups) -- standard deviation of similarity betwenn a group of videos and prompts
    """
    n_total_videos, n_total_prompts = similarities.shape
    n_video_groups = len(video_group_borders) - 1
    n_prompt_groups = len(prompt_group_borders) - 1

    if do_normalize:
        bias = similarities.mean(0, keepdims=True)
        scale = similarities.std(0, keepdims=True)
        similarities = (similarities - bias) / scale

    average_similarities = np.empty((n_video_groups, n_prompt_groups))
    std_similarities = np.empty((n_video_groups, n_prompt_groups))

    for video_group_idx in range(n_video_groups):
        for prompt_group_idx in range(n_prompt_groups):
            p_start, p_end = prompt_group_borders[prompt_group_idx], prompt_group_borders[prompt_group_idx + 1]
            v_start, v_end = video_group_borders[video_group_idx], video_group_borders[video_group_idx + 1]

            average_similarities[video_group_idx, prompt_group_idx] = similarities[v_start:v_end, p_start:p_end].mean()
            std_similarities[video_group_idx, prompt_group_idx] = similarities[v_start:v_end, p_start:p_end].std()

    return average_similarities, std_similarities

def strip_directories_and_extension(path: str) -> str:
    return path.split("/")[-1].split(".")[0]

def compute_projection(xs: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """
    Computes the projections of xs on respective directions.
    Assumes that both xs and directions are unit-length.
        xs: (n_examples, dim)
        directions: (n_examples, dim) or (1, dim)
    """
    coefs = (xs * directions).sum(-1, keepdim=True)
    return coefs * directions

def compute_projection_similarity(prompt_embeds: torch.Tensor, video_embeds: torch.Tensor, directions: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Computes (and mixes) the similarity between prompt embeddings and video embeddings in original and projected space.
    Assumes all mebddings are unit-length.
        prompt_embeds: (n_prompts, dim)
        video_embeds: (n_videos, dim)
        directions: (n_prompts, dim) or (1, dim) -- directions, defining projected 1-d subspaces.
        alpha: float -- mixing coefficient. 0 means no projection.
    """
    orig_sim = video_embeds @ prompt_embeds.T
    proj_sim = video_embeds @ directions.T
    return (1 - alpha) * orig_sim + alpha * proj_sim

def compute_orig_reward(prompt_embeds: torch.Tensor, video_embeds: torch.Tensor, directions: torch.Tensor, alpha: float) -> torch.Tensor:
    """Computes reward as implemented in code of VLM-RM paper (Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning).
    Assumes all mebddings are unit-length.
        prompt_embeds: (n_prompts, dim)
        video_embeds: (n_videos, dim)
        directions: (n_prompts, dim) or (1, dim) -- directions, defining projected 1-d subspaces.
        alpha: float -- mixing coefficient. 0 means no projection.
    """
    mixed_prompt_embeds = (1 - alpha) * prompt_embeds + alpha * compute_projection(prompt_embeds, directions)

    rewards = torch.empty(video_embeds.shape[0], prompt_embeds.shape[0])

    for prompt_idx, direction in enumerate(directions):
        mixed_video_embeds = (1 - alpha) * video_embeds + alpha * compute_projection(video_embeds, direction)
        rewards[:, prompt_idx] = 1 - (mixed_video_embeds - mixed_prompt_embeds[prompt_idx]).square().sum(-1) / 2

    return rewards