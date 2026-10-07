from types import SimpleNamespace

import pytest

from xfuser.model_executor.models.runner_models.cosmos3 import xFuserCosmos3SuperModel


class _RecordingPipe:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(video=["video"])


def _runner():
    model = object.__new__(xFuserCosmos3SuperModel)
    model.pipe = _RecordingPipe()
    model._make_generator = lambda seed: None
    return model


def _input_args(**overrides):
    args = {
        "prompt": "a cat",
        "negative_prompt": "blurry",
        "height": 64,
        "width": 64,
        "num_frames": 5,
        "num_inference_steps": 2,
        "guidance_scale": 6.0,
        "seed": 0,
    }
    args.update(overrides)
    return args


@pytest.mark.parametrize(
    "overrides",
    [
        {"prompt": ["a cat", "a dog"]},
        {"negative_prompt": ["blurry", "dark"]},
    ],
)
def test_more_than_one_prompt_per_call_is_rejected(overrides):
    runner = _runner()

    with pytest.raises(ValueError, match="one .*prompt per pipeline call"):
        runner._run_pipe(_input_args(**overrides))

    assert runner.pipe.calls == []


def test_single_prompt_list_is_unwrapped():
    runner = _runner()

    runner._run_pipe(_input_args(prompt=["a cat"], negative_prompt=["blurry"]))

    assert runner.pipe.calls[0]["prompt"] == "a cat"
    assert runner.pipe.calls[0]["negative_prompt"] == "blurry"
