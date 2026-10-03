"""All-gather accepts tensor views produced by batched CFG inference."""

import pytest

pytestmark = pytest.mark.multi_gpu


def _all_gather_worker(rank, world_size, init_method, layout):
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    from xfuser.core.distributed.group_coordinator import GroupCoordinator

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, init_method=init_method, timeout=timedelta(seconds=30)
    )
    try:
        group = GroupCoordinator([list(range(world_size))], rank, "nccl")
        base = torch.arange(256, dtype=torch.float32, device=device).reshape(2, 8, 4, 4)
        if layout == "cfg_slice":
            reference = base[:, :4]
        elif layout == "transpose":
            reference = base.transpose(1, 2)
        else:
            reference = base
        # Apply the rank offset before taking the view, preserving its strides.
        storage = base + rank * 1000
        if layout == "cfg_slice":
            value = storage.chunk(2, dim=1)[0]
        elif layout == "transpose":
            value = storage.transpose(1, 2)
        else:
            value = storage
        assert value.is_contiguous() == (layout == "contiguous")
        original_storage = storage.clone()
        original_stride = value.stride()
        expected = [reference + source_rank * 1000 for source_rank in range(world_size)]

        for separate in (False, True):
            output = group.all_gather(value, dim=0, separate_tensors=separate)
            if separate:
                assert isinstance(output, list)
                assert len(output) == world_size
                for actual, wanted in zip(output, expected):
                    torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
            else:
                torch.testing.assert_close(output, torch.cat(expected, dim=0), rtol=0, atol=0)
            torch.testing.assert_close(storage, original_storage, rtol=0, atol=0)
            assert value.stride() == original_stride
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("layout", ["contiguous", "cfg_slice", "transpose"])
def test_all_gather_tensor_views(layout, accelerator_ranks):
    accelerator_ranks(_all_gather_worker, world_size=2, init_filename=f"gather-{layout}", args=(layout,))
