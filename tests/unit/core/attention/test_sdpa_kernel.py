"""The aten-backed SDPA kernels compute a log-sum-exp only when the call asks.

The memory-efficient and cuDNN aten ops compute one only on request, and only
ring attention reads it. These tests replace the aten ops at the boundary and
check what the kernels ask for and what they hand back.
"""

from unittest import mock

import pytest
import torch

from xfuser.core.attention.backends.sdpa import kernel
from xfuser.core.attention.spec import AttnCall

_OPS = {
    kernel.sdpa_efficient: "_scaled_dot_product_efficient_attention",
    kernel.cudnn: "_scaled_dot_product_cudnn_attention",
}


def _fake_aten(op_name):
    """An aten stand-in whose op returns (output, lse) the way the real one
    lays them out: the efficient kernel's lse is (B, H, S), cuDNN's (B, H, S, 1)."""
    output = torch.zeros(1, 2, 4, 8)
    lse = torch.ones(1, 2, 4, 1) if op_name.endswith("cudnn_attention") else torch.ones(1, 2, 4)
    op = mock.Mock(return_value=(output, lse, None, None))
    return mock.Mock(**{op_name: op}), op, output


@pytest.mark.parametrize("kernel_fn", list(_OPS), ids=lambda fn: fn.__name__)
def test_kernel_does_not_compute_lse_unless_asked(kernel_fn):
    aten, op, expected_output = _fake_aten(_OPS[kernel_fn])
    qkv = [torch.zeros(1, 2, 4, 8)] * 3
    with mock.patch.object(kernel, "aten", aten):
        output, lse = kernel_fn(*qkv, AttnCall(dropout_p=0.0, is_causal=True))

    assert op.call_args.kwargs["compute_log_sumexp"] is False
    assert op.call_args.kwargs["is_causal"] is True
    assert output is expected_output
    assert lse is None


@pytest.mark.parametrize("kernel_fn", list(_OPS), ids=lambda fn: fn.__name__)
def test_kernel_computes_lse_when_asked(kernel_fn):
    aten, op, expected_output = _fake_aten(_OPS[kernel_fn])
    qkv = [torch.zeros(1, 2, 4, 8)] * 3
    with mock.patch.object(kernel, "aten", aten):
        output, lse = kernel_fn(*qkv, AttnCall(return_lse=True))

    assert op.call_args.kwargs["compute_log_sumexp"] is True
    assert output is expected_output
    assert lse.shape == (1, 2, 4)
