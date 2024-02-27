import os
import imageio.v3 as iio
from typing import List, Callable

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_video(path: str):
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

def uniformly_sample_n_frames(video, n_frames):
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

def make_barplots(average_similarities, std_similarities, video_dir_paths, prompt_group_names, experiment_id, result_dir):
    n_cols = 2
    n_rows = (len(video_dir_paths) + n_cols - 1) // n_cols
    plt.figure(figsize=(12, 3 + 3 * n_rows))
    plt.axis("off")

    for i, dir_path in enumerate(video_dir_paths):
        indices = range(len(prompt_group_names))
        plt.subplot(n_rows, n_cols, i + 1)
        plt.title(dir_path.split('/')[-1])
        plt.bar(indices, average_similarities[i])
        plt.errorbar(indices, average_similarities[i], yerr=std_similarities[i], fmt="o", color="r")
        plt.xticks(indices, labels=prompt_group_names, rotation=30, ha="right")
    
    plt.suptitle(experiment_id)
    plt.tight_layout()
    plt.savefig(f"{result_dir}/aggregated_similarities.png", dpi=350)

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

def strip_directories_and_extension(path: str):
    return path.split("/")[-1].split(".")[0]
