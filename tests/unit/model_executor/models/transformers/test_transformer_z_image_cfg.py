"""CFG parallelism in the xDiT Z-Image transformer needs the pipeline's doubled batch.

ZImagePipeline only batches the negative prompt with the positive one when
``guidance_scale > 0``. With CFG parallelism enabled and guidance turned off the
transformer sees an unsplittable batch, which must be reported as a configuration
error rather than surfacing as an ``IndexError`` deep in the forward pass.
"""

import pytest
import torch

from xfuser.model_executor.models.transformers import transformer_z_image
from xfuser.model_executor.models.transformers.transformer_z_image import (
    xFuserZImageTransformer2DWrapper,
)


@pytest.fixture
def cfg_parallel_transformer(monkeypatch):
    monkeypatch.setattr(transformer_z_image, "get_classifier_free_guidance_world_size", lambda: 2)
    monkeypatch.setattr(transformer_z_image, "get_classifier_free_guidance_rank", lambda: 0)
    monkeypatch.setattr(transformer_z_image, "get_sequence_parallel_world_size", lambda: 1)
    monkeypatch.setattr(transformer_z_image, "get_sequence_parallel_rank", lambda: 0)
    return xFuserZImageTransformer2DWrapper(
        in_channels=4,
        dim=16,
        n_layers=1,
        n_refiner_layers=1,
        n_heads=2,
        n_kv_heads=2,
        cap_feat_dim=8,
        axes_dims=[2, 2, 4],
        axes_lens=[64, 32, 32],
    )


def test_cfg_parallel_without_doubled_batch_raises_value_error(cfg_parallel_transformer):
    x = [torch.randn(4, 1, 4, 4)]
    cap_feats = [torch.randn(3, 8)]
    t = torch.tensor([0.5])

    with pytest.raises(ValueError, match="guidance_scale > 0"):
        cfg_parallel_transformer(x, t, cap_feats)
