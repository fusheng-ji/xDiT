"""On a device, the efficient and cuDNN SDPA kernels skip the log-sum-exp
unless the call asks for it, and return a correct one when it does."""

import pytest
import torch
import torch.nn.functional as F

from xfuser.core.attention.backends.sdpa import kernel
from xfuser.core.attention.spec import AttnCall

KERNELS = [
    pytest.param(kernel.sdpa_efficient, torch.float32, id="efficient-fp32"),
    pytest.param(kernel.sdpa_efficient, torch.bfloat16, id="efficient-bf16"),
    pytest.param(kernel.cudnn, torch.bfloat16, id="cudnn-bf16"),
]


def _qkv(dtype, seq_len=64):
    generator = torch.Generator().manual_seed(0)
    return [torch.randn(1, 2, seq_len, 64, generator=generator).to(device="cuda", dtype=dtype) for _ in range(3)]


def _run(kernel_fn, query, key, value, call):
    try:
        return kernel_fn(query, key, value, call)
    except RuntimeError as error:
        pytest.skip(f"{kernel_fn.__name__} is unavailable here: {error}")


@pytest.mark.parametrize("kernel_fn, dtype", KERNELS)
def test_kernel_not_asked_for_lse_matches_sdpa_and_returns_none(kernel_fn, dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator")
    query, key, value = _qkv(dtype)
    output, lse = _run(kernel_fn, query, key, value, AttnCall())

    assert lse is None
    expected = F.scaled_dot_product_attention(query.float(), key.float(), value.float())
    torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("kernel_fn, dtype", KERNELS)
def test_kernel_asked_for_lse_returns_it(kernel_fn, dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator")
    query, key, value = _qkv(dtype)
    output, lse = _run(kernel_fn, query, key, value, AttnCall(return_lse=True))

    scores = query.float() @ key.float().transpose(-1, -2) * query.shape[-1] ** -0.5
    expected_lse = torch.logsumexp(scores, dim=-1)
    torch.testing.assert_close(lse.float(), expected_lse, rtol=2e-2, atol=2e-2)
    expected = F.scaled_dot_product_attention(query.float(), key.float(), value.float())
    torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=2e-2)
