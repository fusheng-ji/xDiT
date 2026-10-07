"""Request validation for the causal Wan pipeline, on CPU without weights."""

import pytest
import torch

from xfuser.model_executor.pipelines.pipeline_causal_wan import xFuserCausalWanPipeline


class _ReachedGeneration(Exception):
    pass


def _pipeline():
    pipe = object.__new__(xFuserCausalWanPipeline)
    pipe.vae_scale_factor_temporal = 4
    pipe._callback_tensor_inputs = ["latents"]
    return pipe


def _run(pipe, monkeypatch, **kwargs):
    # Stop right after request validation, before any model is needed.
    def stop(*args, **kwargs):
        raise _ReachedGeneration

    monkeypatch.setattr(xFuserCausalWanPipeline, "config", property(stop), raising=False)
    pipe(prompt_embeds=torch.zeros(1, 4, 8), height=16, width=16, guidance_scale=0.0, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_frames": 93},
        {"num_frames": 49, "sliding_window_num_frames": 12},
        {"num_frames": 81, "latents": torch.zeros(1, 4, 24, 2, 2)},
    ],
    ids=["default-window", "smaller-window", "latents-provided"],
)
def test_request_longer_than_kv_cache_is_rejected(monkeypatch, kwargs):
    with pytest.raises(ValueError, match="sliding_window_num_frames"):
        _run(_pipeline(), monkeypatch, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"num_frames": 81}, {"num_frames": 93, "local_attn_size": 12}],
    ids=["fits-cache", "sliding-window"],
)
def test_request_within_kv_cache_capacity_is_accepted(monkeypatch, kwargs):
    with pytest.raises(_ReachedGeneration):
        _run(_pipeline(), monkeypatch, **kwargs)
