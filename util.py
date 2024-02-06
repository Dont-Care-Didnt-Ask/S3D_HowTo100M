import imageio.v3 as iio

def load_video(path: str):
    return iio.imread(path, plugin="pyav")

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
