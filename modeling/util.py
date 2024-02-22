import imageio.v3 as iio
from typing import List

def load_video(path: str):
    if path.endswith(".mp4"):
        return iio.imread(path, plugin="pyav")
    elif path.endswith(".avi"):
        return iio.imread(path, format="FFMPEG")

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
