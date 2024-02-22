import pytest
from typing import List

import numpy as np

from modeling.util import load_prompts, load_video

#content of some_prompts.txt
SOME_PROMPTS_TXT_CONTENT = """A stick model of a dog running
A stick model of a dog running on back legs
A stick model of a dog running in small steps
A stick model of a dog falling and crawling
"""

@pytest.fixture
def path_to_prompts(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir()
    p = d / "some_prompts.txt"
    p.write_text(SOME_PROMPTS_TXT_CONTENT)
    return p

@pytest.mark.parametrize(["verbose"],[[True], [False]])
def test_load_promts(path_to_prompts: str, verbose: bool):
    prompts = load_prompts(path_to_prompts, verbose)
    assert isinstance(prompts, List)
    assert all(isinstance(p, str) for p in prompts)

def test_load_video(path_to_video: str = "imageio:cockatoo.mp4"):
    video = load_video(path_to_video)
    assert isinstance(video, np.ndarray)