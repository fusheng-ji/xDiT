"""A per-step hybrid schedule must start at its first step on every pipeline
run, and Krea-2's schedule must span both guidance passes of a step."""

from types import SimpleNamespace

import pytest
import torch

from xfuser.core.attention.spec import AttentionBackendType
from xfuser.core.distributed.attention_schedule import AttentionSchedule
from xfuser.core.distributed.runtime_state import DiTRuntimeState
from xfuser.model_executor.models.runner_models import base_model, krea2
from xfuser.model_executor.models.runner_models.base_model import DiffusionOutput, xFuserModel

HIGH = AttentionBackendType.SDPA
LOW = AttentionBackendType.SDPA_MATH


def _scheduled_state(backends):
    state = DiTRuntimeState.__new__(DiTRuntimeState)
    state.attention_schedule = AttentionSchedule(backends)
    state.schedule_total_steps = torch.tensor(len(backends), dtype=torch.int)
    state.gemm_schedule = None
    state.gemm_schedule_total_steps = None
    state.step_counter = torch.tensor(0, dtype=torch.int)
    state.attention_backend = HIGH
    return state


class _Event:
    def __init__(self, enable_timing=True):
        pass

    def record(self):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return 0.0


class _Runner(xFuserModel):
    """Each pipeline run calls the transformer once per step, recording the
    attention backend the schedule selects for that call."""

    def _load_model(self):
        raise NotImplementedError

    def _run_pipe(self, input_args):
        seen = []
        for _ in range(input_args["num_inference_steps"]):
            self.state.increment_step_counter()
            seen.append(self.state.attention_backend)
        return DiffusionOutput(images=[], pipe_args=input_args), seen


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(base_model, "get_model_replica_group", lambda: SimpleNamespace(barrier=lambda: None))

    model = object.__new__(_Runner)
    model.state = _scheduled_state([HIGH, LOW, LOW, HIGH])
    model._vae_manager = SimpleNamespace(prepare_run=lambda vaes, input_args: None)
    model._decoding_vaes = lambda: []
    monkeypatch.setattr(base_model, "get_runtime_state", lambda: model.state)
    return model


@pytest.mark.parametrize("warmup_steps", [1, 2, 3])
def test_run_after_a_shorter_warmup_follows_the_schedule_from_its_first_step(runner, warmup_steps):
    runner._run_timed_pipe({"num_inference_steps": warmup_steps})

    (_, seen), _ = runner._run_timed_pipe({"num_inference_steps": 4})

    assert seen == [HIGH, LOW, LOW, HIGH]


class _ScheduleConfig(SimpleNamespace):
    hybrid_attn_low_precision_backend = "sdpa_math"
    hybrid_attn_high_precision_backend = "sdpa"
    hybrid_attn_schedule = None


@pytest.mark.parametrize(
    ("model_cls", "guidance_scale", "transformer_calls_per_step"),
    [
        (krea2.xFuserKrea2RawModel, 3.5, 2),
        (krea2.xFuserKrea2TurboModel, 0.0, 1),
    ],
    ids=["raw-guided", "turbo"],
)
def test_krea2_schedule_spans_every_transformer_call(
    monkeypatch, model_cls, guidance_scale, transformer_calls_per_step
):
    state = SimpleNamespace(_check_if_backend_compatible_with_current_configuration=lambda backend: None)
    scheduled = {}
    state.set_attention_schedule = lambda schedule, total_steps: scheduled.update(
        backends=schedule.backends, total_steps=total_steps
    )
    monkeypatch.setattr(base_model, "get_runtime_state", lambda: state)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(base_model, "get_world_group", lambda: SimpleNamespace(rank=0), raising=False)

    model = object.__new__(model_cls)
    model.config = _ScheduleConfig()
    model._setup_hybrid_attn_schedule(
        {"num_inference_steps": 8, "guidance_scale": guidance_scale, "num_hybrid_attn_high_precision_steps": 1}
    )

    calls = 8 * transformer_calls_per_step
    assert scheduled["total_steps"] == calls
    # One high-precision denoising step at each end covers all of its calls.
    high = [HIGH] * transformer_calls_per_step
    assert scheduled["backends"] == high + [LOW] * (calls - 2 * transformer_calls_per_step) + high
