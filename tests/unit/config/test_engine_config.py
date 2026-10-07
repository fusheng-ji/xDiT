import pytest
import torch

from xfuser.config.config import (
    DataParallelConfig,
    EngineConfig,
    FastAttnConfig,
    FullyShardConfig,
    ModelConfig,
    ParallelConfig,
    PipeFusionParallelConfig,
    RuntimeConfig,
    SequenceParallelConfig,
    TensorParallelConfig,
    VaeParallelConfig,
)


def _engine_config(pp_degree: int, warmup_steps: int) -> EngineConfig:
    return EngineConfig(
        model_config=ModelConfig(model="model"),
        runtime_config=RuntimeConfig(warmup_steps=warmup_steps, dtype=torch.float32),
        parallel_config=ParallelConfig(
            dp_config=DataParallelConfig(dit_parallel_size=pp_degree),
            sp_config=SequenceParallelConfig(dit_parallel_size=pp_degree),
            pp_config=PipeFusionParallelConfig(
                pp_degree=pp_degree, attn_layer_num_for_pp=None, dit_parallel_size=pp_degree
            ),
            tp_config=TensorParallelConfig(dit_parallel_size=pp_degree),
            fs_config=FullyShardConfig(),
            vae_config=VaeParallelConfig(),
            world_size=pp_degree,
            dit_parallel_size=pp_degree,
        ),
        fast_attn_config=FastAttnConfig(),
    )


def test_pipefusion_rejects_zero_warmup_steps():
    with pytest.raises(ValueError, match="warmup_steps must be at least 1"):
        _engine_config(pp_degree=2, warmup_steps=0)


@pytest.mark.parametrize(("pp_degree", "warmup_steps"), [(2, 1), (1, 0)])
def test_warmup_steps_accepted_when_pipefusion_can_fill_its_cache(pp_degree, warmup_steps):
    config = _engine_config(pp_degree=pp_degree, warmup_steps=warmup_steps)

    assert config.runtime_config.warmup_steps == warmup_steps
