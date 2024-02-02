import argparse
import imageio.v3 as iio

import torch
import numpy as np
import torch.nn.functional as F

from s3dg import S3D

def parse_args():
    parser = argparse.ArgumentParser(description="Compare trajectory with given prompts")

    parser.add_argument("-v", "--video-path", help="Path to trajectory in mp4 format.", required=True)
    parser.add_argument("-p", "--prompts-path", help="Path to prompts in txt format. Expected to have one prompt per line.", required=True)
    parser.add_argument("--n-frames", type=int, default=32)

    args = parser.parse_args()
    return args

def load_video(path: str):
    return iio.imread(path, plugin="pyav")

def prepare_video(video: np.ndarray, n_frames: int, verbose: bool):
    if verbose:
        print("Initial video shape:", video.shape, " dtype:", video.dtype)

    # Probably not most accurate frame sampling -- might 
    length = video.shape[0]
    step_size = length // (n_frames - 1)
    video = video[::step_size][:n_frames]
    
    video = torch.from_numpy(video)

    if video.dtype not in (torch.float16, torch.float32, torch.float64):
        video = video.float() / 255
    
    video = F.interpolate(
        video.permute(0, 3, 1, 2),
        mode="bicubic",
        scale_factor=0.5,
    ).transpose(0,1)

    if verbose:
        print("Min and max before clipping:", video.min(), video.max())

    video = video.clamp(0, 1)

    if verbose:
        print("Final video shape:", video.shape, " dtype:", video.dtype)

    return video.unsqueeze(0)

def load_model():
    # Instantiate the model
    net = S3D('checkpoint/s3d_dict.npy', 512)
    # Load the model weights
    net.load_state_dict(torch.load('checkpoint/s3d_howto100m.pth'))
    # Evaluation mode
    net = net.eval()
    return net

def load_prompts(path: str, verbose: bool):
    prompts = []

    with open(path, "r") as f:
        for line in f.readlines():
            prompts.append(line.rstrip("\n"))


    if verbose:
        print("Loaded promts:")
        for i, p in enumerate(prompts):
            print(f"{i:2d}: {p}")

    return prompts

def main():
    args = parse_args()

    # Video input should be of size Batch x 3 x T x H x W and normalized to [0, 1] 
    # Also, afaik expects either 32 or 16 frames
    video = prepare_video(load_video(args.video_path), n_frames=args.n_frames, verbose=True)
    
    prompts = load_prompts(args.prompts_path, verbose=False)

    net = load_model()
    
    # Video inference
    print("Embedding video...")
    video_output = net(video)
    
    # Text inference
    print("Embedding text...")
    text_output = net.text_module(prompts)

    v_embed = video_output["video_embedding"] / video_output["video_embedding"].norm(p=2, dim=-1, keepdim=True)
    p_embeds = text_output["text_embedding"] / text_output["text_embedding"].norm(p=2, dim=-1, keepdim=True)

    similarities = (v_embed @ p_embeds.T).squeeze()

    for i, prompt in enumerate(prompts):
        print(f"Prompt {i:2d}: {prompt:<50}\tSimilarity: {similarities[i].item():.3f}")


if __name__ == "__main__":
    main()
