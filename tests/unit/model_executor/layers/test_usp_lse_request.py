"""Only ring attention asks the attention kernel for a log-sum-exp.

Ring attention merges each step's partial output on it; every other caller
discards it, so the request travels on the call rather than being implied.
"""

from types import SimpleNamespace

import torch

from xfuser.model_executor.layers import usp


def _recording_attention_function():
    calls = []

    def run(query, key, value, call):
        calls.append(call)
        return query, (torch.zeros(query.shape[:-1]) if call.return_lse else None)

    return usp.concat_joint_tensors_decorator(usp._spec_adapter(SimpleNamespace(run=run))), calls


def test_ring_attention_requests_the_lse(monkeypatch):
    attention_function, calls = _recording_attention_function()

    def one_step_ring(*args, **kwargs):
        op, query, key, value = args[-4:]
        return op(query, key, value, **kwargs)

    monkeypatch.setattr(usp, "_templated_ring_attention", one_step_ring)
    query = torch.randn(1, 2, 4, 8)

    usp.ring_attn(attention_function, query, query, query)

    assert [call.return_lse for call in calls] == [True]


def test_a_direct_call_does_not_request_the_lse():
    attention_function, calls = _recording_attention_function()
    query = torch.randn(1, 2, 4, 8)

    _, lse = attention_function(query, query, query, dropout_p=0.0, is_causal=False)

    assert [call.return_lse for call in calls] == [False]
    assert lse is None
