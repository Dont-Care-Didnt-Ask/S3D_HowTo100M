import argparse

import torch
import numpy as np
import torch.nn.functional as F

from modeling.s3dg import S3D
from modeling.util import load_prompts, load_video

def parse_args():
    parser = argparse.ArgumentParser(description="Compare trajectory with given prompts")

    parser.add_argument("-t", "--trajectory-path", help="Path to trajectory in mp4 format.", required=True)
    parser.add_argument("-p", "--prompts-path", help="Path to prompts in txt format. Expected to have one prompt per line.", required=True)
    parser.add_argument("--n-frames", type=int, default=32)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--model-checkpoint-path", default="checkpoints/s3d_howto100m.pth")

    args = parser.parse_args()
    return args

def prepare_video(video: np.ndarray, n_frames: int, verbose: bool) -> torch.Tensor:
    if verbose:
        print("Initial video shape:", video.shape, " dtype:", video.dtype)

    # Probably not most accurate frame sampling -- might be improved
    length = video.shape[0]
    step_size = length // n_frames
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

    video = video.clamp(0, 1).unsqueeze(0)

    if verbose:
        print("Final video shape:", video.shape, " dtype:", video.dtype)

    return video

def load_model(model_checkpoint_path):
    # Instantiate the model
    embedding_dim = 512
    net = S3D('checkpoints/s3d_dict.npy', embedding_dim)
    # Load the model weights
    net.load_state_dict(torch.load(model_checkpoint_path))
    # Evaluation mode
    net = net.eval()
    return net

@torch.inference_mode()
def main():
    args = parse_args()
    if args.verbose:
        print(f"Running S3D evaluator with following args:\n{args}")

    # Video input should be of size Batch x 3 x T x H x W and normalized to [0, 1] 
    # Also, afaik expects either 32 or 16 frames
    video = prepare_video(load_video(args.trajectory_path), n_frames=args.n_frames, verbose=False)
    
    prompts = load_prompts(args.prompts_path, verbose=False)

    net = load_model(args.model_checkpoint_path)
    
    # Video inference
    if args.verbose:
        print("Embedding video...")
    video_output = net(video)
    
    # Text inference
    if args.verbose:
        print("Embedding text...")
    text_output = net.text_module(prompts)

    v_embed = video_output["video_embedding"] / video_output["video_embedding"].norm(p=2, dim=-1, keepdim=True)
    p_embeds = text_output["text_embedding"] / text_output["text_embedding"].norm(p=2, dim=-1, keepdim=True)

    similarities = (v_embed @ p_embeds.T).squeeze()

    for i, prompt in enumerate(prompts):
        print(f"Prompt {i:2d}: {prompt:<70}\tSimilarity: {similarities[i].item():.3f}")


if __name__ == "__main__":
    main()
