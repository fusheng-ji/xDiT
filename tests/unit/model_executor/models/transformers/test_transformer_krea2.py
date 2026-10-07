"""The Krea-2 wrapper drives the per-step hybrid attention schedule."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from xfuser.core.attention.spec import AttentionBackendType
from xfuser.core.distributed import runtime_state as runtime_state_module
from xfuser.core.distributed.attention_schedule import AttentionSchedule
from xfuser.core.distributed.runtime_state import DiTRuntimeState
from xfuser.model_executor.models.transformers import transformer_krea2

HIGH = AttentionBackendType.SDPA
LOW = AttentionBackendType.SDPA_MATH


def _attention(query, key, value, **_):
    return F.scaled_dot_product_attention(query, key, value)


@pytest.fixture
def scheduled_state(monkeypatch):
    state = DiTRuntimeState.__new__(DiTRuntimeState)
    state.attention_schedule = AttentionSchedule([HIGH, LOW, HIGH])
    state.schedule_total_steps = torch.tensor(3, dtype=torch.int)
    state.gemm_schedule = None
    state.gemm_schedule_total_steps = None
    state.step_counter = torch.tensor(0, dtype=torch.int)
    state.attention_backend = LOW
    monkeypatch.setattr(runtime_state_module, "_RUNTIME", state)
    return state


@pytest.fixture
def krea2():
    torch.manual_seed(0)
    model = transformer_krea2.xFuserKrea2Transformer2DWrapper(
        in_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=16,
        timestep_embed_dim=8,
        text_hidden_dim=8,
        num_text_layers=2,
        text_num_attention_heads=2,
        text_num_key_value_heads=2,
        text_intermediate_size=16,
        num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1,
        axes_dims_rope=(2, 2, 4),
    ).eval()
    with (
        mock.patch.object(transformer_krea2, "get_sequence_parallel_world_size", return_value=1),
        mock.patch.object(transformer_krea2, "get_sequence_parallel_rank", return_value=0),
        mock.patch(
            "xfuser.model_executor.models.transformers.transformers_utils.get_sp_group",
            return_value=SimpleNamespace(all_gather=lambda x, dim: x),
        ),
        mock.patch.object(transformer_krea2, "USP", _attention),
    ):
        yield model


@torch.no_grad()
def test_each_forward_selects_the_next_scheduled_backend(krea2, scheduled_state):
    torch.manual_seed(1)
    inputs = dict(
        hidden_states=torch.randn(1, 4, 4),
        encoder_hidden_states=torch.randn(1, 6, 2, 8),
        timestep=torch.tensor([0.5]),
        position_ids=torch.randint(0, 4, (10, 3)),
    )

    seen = []
    for _ in range(3):
        krea2(**inputs)
        seen.append(scheduled_state.attention_backend)

    assert seen == [HIGH, LOW, HIGH]
