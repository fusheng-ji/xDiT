"""Data parallelism splits the SDXL prompt batch even when no other parallelism is on.

With only data parallelism, the wrapper takes the naive forward path. The prompt
split has to happen before that, or every data-parallel rank renders the whole batch.
"""

from types import SimpleNamespace

from xfuser.model_executor.pipelines import base_pipeline
from xfuser.model_executor.pipelines.pipeline_stable_diffusion_xl import (
    xFuserStableDiffusionXLPipeline,
)


class _RecordingModule:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return "output"


def _data_parallel_only(monkeypatch, dp_degree, rank):
    parallel_config = SimpleNamespace(dp_degree=dp_degree, cfg_degree=1, sp_degree=1, pp_degree=1, vae_parallel_size=0)
    monkeypatch.setattr(base_pipeline, "get_runtime_state", lambda: SimpleNamespace(parallel_config=parallel_config))
    monkeypatch.setattr(base_pipeline, "get_world_group", lambda: SimpleNamespace(rank=rank))
    monkeypatch.setattr(base_pipeline, "get_dit_world_size", lambda: dp_degree)
    monkeypatch.setattr(base_pipeline, "get_data_parallel_world_size", lambda: dp_degree)
    for name in (
        "get_pipeline_parallel_world_size",
        "get_classifier_free_guidance_world_size",
        "get_sequence_parallel_world_size",
        "get_tensor_model_parallel_world_size",
    ):
        monkeypatch.setattr(base_pipeline, name, lambda: 1)
    monkeypatch.setattr(base_pipeline, "get_fast_attn_enable", lambda: False)


def test_data_parallel_only_rank_receives_its_prompt_share(monkeypatch):
    _data_parallel_only(monkeypatch, dp_degree=2, rank=1)
    pipeline = object.__new__(xFuserStableDiffusionXLPipeline)
    pipeline.module = _RecordingModule()

    pipeline(prompt=["a", "b", "c", "d"], negative_prompt=["na", "nb", "nc", "nd"])

    assert pipeline.module.calls == [{"prompt": ["c", "d"], "negative_prompt": ["nc", "nd"]}]
