# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3 model dispatch regression for retained BF16 prefill with FP8 storage.

Execute current source methods on CPU; substitute only native cache insertion,
attention execution and device discovery. Native insertion is a separate gate.
"""

import ast
import functools
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(os.environ.get("VLLM_SOURCE_ROOT") or Path(__file__).resolve().parents[3])
MLA = "vllm/models/kimi_k3/nvidia/mla.py"
COMMON = "vllm/model_executor/layers/attention/mla_attention.py"


def _method(path, name, namespace, parent=None):
    nodes = ast.parse((ROOT / path).read_text()).body
    if parent:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == parent
        ).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace
    )
    return namespace[name]


def _functions(insert, quant_insert=None):
    ns = dict(
        torch=torch,
        functools=functools,
        logger=Mock(),
        current_platform=SimpleNamespace(
            fp8_dtype=lambda: torch.float8_e4m3fn,
            is_device_capability_family=lambda family: family == 120,
        ),
        is_quantized_kv_cache=lambda dtype: dtype.startswith("fp8"),
        ops=SimpleNamespace(concat_and_cache_mla=insert),
        fused_mla_qkv_quant_kv_cache_fp8_insert=quant_insert,
        merge_attn_states=lambda **kw: kw["output"].copy_(
            kw["prefix_output"] + kw["suffix_output"]
        ),
    )
    _method(COMMON, "backend_supports_prefill_query_quantization", ns)
    choose_dtype = _method(
        COMMON, "determine_prefill_query_data_type", ns, "MLACommonMetadataBuilder"
    )
    forward = _method(MLA, "_forward_prefill_fused", ns, "MultiHeadLatentAttention")
    return choose_dtype, forward


@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3"])
@pytest.mark.parametrize("honors_out", [False, True])
@pytest.mark.parametrize("dcp_context", [False, True])
def test_bf16_prefill_preserves_query_and_scales_only_cache(
    cache_dtype, honors_out, dcp_context
):
    torch.manual_seed(17)
    tokens, heads, latent, nope, rope, value = 4, 6, 512, 128, 64, 128
    q = torch.randn(tokens, heads, nope + rope).bfloat16()
    original_q = q.clone()
    kv = torch.randn(tokens, latent).bfloat16()
    k_pe = torch.randn(tokens, 1, rope).bfloat16()
    projected = torch.randn(tokens, heads * (nope + value)).bfloat16()
    cache = torch.zeros(2, 16, latent + rope, dtype=torch.float8_e4m3fn)
    slots = torch.tensor([0, 17, -1, 5], dtype=torch.int64)
    scale = torch.tensor([0.5], dtype=torch.float32)
    calls = []

    def insert(*args, **kwargs):
        calls.append((args, kwargs))

    choose_dtype, forward = _functions(
        insert,
        quant_insert=Mock(
            side_effect=AssertionError("Q/K/V quantization must not run")
        ),
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        attention_config=SimpleNamespace(use_prefill_query_quantization=False),
    )
    q_dtype = choose_dtype(config, torch.bfloat16)
    assert q_dtype == torch.bfloat16
    observed = []
    expected_k_nope, expected_v = projected.view(tokens, heads, nope + value).split(
        [nope, value], dim=-1
    )
    expected_k = torch.cat((expected_k_nope, k_pe.expand(-1, heads, -1)), dim=-1)
    result = torch.full((tokens, heads, value), 2, dtype=torch.bfloat16)

    def new_tokens(**kwargs):
        observed.append(kwargs)
        for actual, expected in (
            (kwargs["q"], original_q),
            (kwargs["k"], expected_k),
            (kwargs["v"], expected_v),
        ):
            assert actual.dtype == torch.bfloat16
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert kwargs["q"] is q
        if kwargs["out"] is not None:
            kwargs["out"].copy_(result)
            return kwargs["out"]
        return (result, torch.zeros(heads, tokens)) if dcp_context else result

    context_calls = []

    def context(*args, **kwargs):
        context_calls.append((args, kwargs))
        assert args[0] is q and args[0].dtype == torch.bfloat16
        assert kwargs["dcp_world_size"] == 4 and kwargs["k_scale"] is scale
        return torch.ones_like(result), torch.zeros(heads, tokens)

    layer = SimpleNamespace(
        kv_b_proj=lambda _: (projected, None),
        num_local_heads=heads,
        qk_nope_head_dim=nope,
        v_head_dim=value,
        kv_cache_dtype=cache_dtype,
        kv_cache=cache,
        _k_scale=scale,
        dcp_world_size=4 if dcp_context else 1,
        _attn_read_kv_cache=lambda: cache,
        _fused_mla_kv_concat=object(),
        impl=SimpleNamespace(_context_parallel_compute_prefill_context=context),
    )
    metadata = SimpleNamespace(
        prefill=SimpleNamespace(
            q_data_type=q_dtype,
            chunked_context=object() if dcp_context else None,
            prefill_backend=SimpleNamespace(
                supports_out=lambda: honors_out, run_prefill_new_tokens=new_tokens
            ),
        )
    )
    out = torch.empty(tokens, heads * value, dtype=torch.bfloat16)
    forward(layer, q, kv, k_pe, None, None, slots, metadata, out)
    assert len(calls) == len(observed) == 1
    args, kwargs = calls[0]
    assert args[0] is kv and args[2] is cache
    torch.testing.assert_close(args[1], k_pe.flatten(1), atol=0, rtol=0)
    torch.testing.assert_close(args[3], slots, atol=0, rtol=0)
    assert kwargs == dict(kv_cache_dtype=cache_dtype, scale=scale)
    assert len(context_calls) == int(dcp_context)
    torch.testing.assert_close(
        out, result.flatten(1) + int(dcp_context), atol=0, rtol=0
    )
    torch.testing.assert_close(q, original_q, atol=0, rtol=0)


def test_enabling_fp8_query_flag_does_not_bypass_sm121_backend_gate():
    choose_dtype, _ = _functions(Mock())
    config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="fp8"),
        attention_config=SimpleNamespace(use_prefill_query_quantization=True),
    )
    assert choose_dtype(config, torch.bfloat16) == torch.bfloat16


def test_restored_bf16_fallback_still_rejects_rope():
    _, forward = _functions(Mock())
    layer = SimpleNamespace(
        kv_b_proj=lambda _: (torch.ones(2, 24).bfloat16(), None),
        num_local_heads=2,
        qk_nope_head_dim=6,
        v_head_dim=6,
        kv_cache_dtype="fp8",
    )
    metadata = SimpleNamespace(
        prefill=SimpleNamespace(q_data_type=torch.bfloat16, chunked_context=None)
    )
    with pytest.raises(AssertionError, match="NoPE"):
        forward(
            layer,
            torch.ones(2, 2, 8).bfloat16(),
            torch.ones(2, 4).bfloat16(),
            torch.ones(2, 2).bfloat16(),
            torch.arange(2),
            None,
            torch.arange(2),
            metadata,
            torch.empty(2, 12).bfloat16(),
        )
