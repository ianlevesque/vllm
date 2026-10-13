# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU oracle and actual MiMo callers; no vLLM/CUDA initialization.

Run with pytest --confcutdir=tests/standalone. External kernel/projection boundaries
are injected. Native arithmetic and real TP communication remain separate gates.
"""

import math
import sys
from types import MethodType, ModuleType, SimpleNamespace

import einops
import pytest
import torch
from mimo_vision_sink_reference import (
    CASES,
    HEAD_DIM,
    assert_discriminating,
    dense_reference,
    make_inputs,
)
from test_shared_source_carries import load

SOURCE = "vllm/model_executor/models/mimo_v2_omni.py"
KERNEL = "vllm.v1.attention.ops.triton_prefill_attention"


def test_appended_logit_oracle_has_known_half_output():
    q = k = torch.zeros(2, 1, 1)
    v = torch.ones_like(q)
    cu, sinks = torch.tensor([0, 2]), torch.tensor([math.log(2)])
    for mode, expected in (("denominator", 0.5), ("key0", 1.0), ("none", 1.0)):
        result = dense_reference(q, k, v, cu, sinks, 1, 8, mode)
        torch.testing.assert_close(result, torch.full_like(result, expected))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_frozen_oracle_rejects_both_wrong_semantics_and_cross_image_keys(dtype, case):
    q, k, v, cu, _, sinks, window = make_inputs(case, dtype)
    scale = HEAD_DIM**-0.5
    reference = dense_reference(q, k, v, cu, sinks, scale, window)
    assert_discriminating(q, k, v, cu, sinks, scale, window, reference)
    changed = v.clone()
    changed[: cu[1]] = -20
    isolated = dense_reference(q, k, changed, cu, sinks, scale, window)
    torch.testing.assert_close(isolated[cu[1] :], reference[cu[1] :], rtol=0, atol=0)
    flattened = dense_reference(
        q, k, v, torch.tensor([0, len(q)]), sinks, scale, window
    )
    assert not torch.allclose(flattened, reference, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
@pytest.mark.parametrize("use_sink", [True, False])
def test_actual_window_caller_slices_heads_and_preserves_packing(
    monkeypatch, case, use_sink
):
    q, k, v, cu, global_sinks, sinks, window = make_inputs(case, torch.bfloat16)
    _, heads, _, tp, rank, _, lengths = case
    owner = SimpleNamespace(
        tp_rank=rank,
        num_heads_per_partition=heads // tp,
        sinks=global_sinks if use_sink else None,
        visual_token_window_size=window,
        scale=HEAD_DIM**-0.5,
    )
    calls = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        args[3].fill_(7)

    module = ModuleType(KERNEL)
    module.context_attention_fwd = capture
    monkeypatch.setitem(sys.modules, KERNEL, module)
    method = load(SOURCE, "MiMoVisionAttention._forward_window_attn")
    result = method(owner, q, k, v, cu, max(lengths))
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is q and args[1] is k and args[2] is v and args[3] is result
    assert kwargs["sinks_bias_key0"] is False
    assert kwargs["is_causal"] is False
    assert kwargs["max_input_len"] == max(lengths)
    assert kwargs["sliding_window_q"] == kwargs["sliding_window_k"] == window
    assert kwargs["softmax_scale"] == owner.scale
    torch.testing.assert_close(kwargs["b_start_loc"], cu[:-1], rtol=0, atol=0)
    torch.testing.assert_close(kwargs["b_seq_len"], cu[1:] - cu[:-1], rtol=0, atol=0)
    if use_sink:
        torch.testing.assert_close(kwargs["sinks"], sinks, rtol=0, atol=0)
        assert kwargs["sinks"].data_ptr() == sinks.data_ptr()
    else:
        assert kwargs["sinks"] is None
    torch.testing.assert_close(result, torch.full_like(q, 7), rtol=0, atol=0)


@pytest.mark.parametrize(
    "full_attention,backend_ndim", [(True, 3), (True, 4), (False, 3)]
)
def test_actual_forward_routes_full_and_window_attention(
    monkeypatch, full_attention, backend_ndim
):
    case = CASES[-1]
    q, k, v, cu, global_sinks, _, window = make_inputs(case, torch.float16)
    total, heads, dim = q.shape
    projected = torch.cat([x.reshape(total, 1, -1) for x in (q, k, v)], dim=-1)
    output = torch.arange(q.numel(), dtype=torch.float32).reshape_as(q).to(q.dtype)
    calls, rotary_calls = [], []

    def rotary(value, cos, sin):
        rotary_calls.append((value.clone(), cos, sin))
        return value + 1

    def full(**kwargs):
        assert full_attention
        calls.append(("full", kwargs))
        result = output.unsqueeze(0)
        return result if backend_ndim == 4 else result.flatten(2)

    def kernel(*args, **kwargs):
        assert not full_attention
        calls.append(("window", (args, kwargs)))
        args[3].copy_(output)

    module = ModuleType(KERNEL)
    module.context_attention_fwd = kernel
    monkeypatch.setitem(sys.modules, KERNEL, module)
    owner = SimpleNamespace(
        num_heads_per_partition=heads,
        num_kv_heads_per_partition=1,
        qk_channels=dim,
        kv_channels=dim,
        tp_rank=7,
        sinks=global_sinks,
        visual_token_window_size=window,
        scale=dim**-0.5,
        qkv=lambda x: (projected, None),
        proj=lambda x: (x, None),
        attn=full,
        apply_rotary_emb=rotary,
    )
    owner._forward_window_attn = MethodType(
        load(SOURCE, "MiMoVisionAttention._forward_window_attn"), owner
    )
    method = load(SOURCE, "MiMoVisionAttention.forward", einops=einops)
    cos, sin = torch.ones(1), torch.zeros(1)
    result = method(
        owner, torch.empty(total, 1, 1280), cu, cos, sin, 137, full_attention
    )
    assert len(calls) == 1 and len(rotary_calls) == 2
    torch.testing.assert_close(rotary_calls[0][0], q.unsqueeze(0))
    torch.testing.assert_close(rotary_calls[1][0], k.unsqueeze(0))
    assert all(c is cos and s is sin for _, c, s in rotary_calls)
    if full_attention:
        kwargs = calls[0][1]
        assert set(kwargs) == {"query", "key", "value", "cu_seqlens", "max_seqlen"}
        for name, expected in (("query", q + 1), ("key", k + 1), ("value", v)):
            torch.testing.assert_close(kwargs[name], expected.reshape(1, total, -1))
        assert kwargs["cu_seqlens"] is cu and kwargs["max_seqlen"] == 137
    else:
        args, kwargs = calls[0][1]
        for actual, expected in zip(args[:3], (q + 1, k + 1, v)):
            torch.testing.assert_close(actual, expected)
        assert kwargs["sinks_bias_key0"] is False
        torch.testing.assert_close(kwargs["sinks"], global_sinks[28:32])
    torch.testing.assert_close(result, output.reshape(total, 1, -1), rtol=0, atol=0)
