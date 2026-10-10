# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU numerical gates for retained Kimi storage and TP geometry.

Load the actual pure functions/methods from source so these can run without
CUDA extensions; GPU/graph/distributed qualification remains a separate gate.
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(os.environ.get("VLLM_SOURCE_ROOT") or Path(__file__).resolve().parents[3])
KIMI = "vllm/models/kimi_k3/nvidia/"


def load_source(relative, names, *, parent=None, **namespace):
    path = ROOT / relative
    nodes = ast.parse(path.read_text()).body
    if parent:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == parent
        ).body
    selected = [
        n
        for n in nodes
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    assert len(selected) == len(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    namespace = dict(torch=torch, nn=torch.nn, logger=Mock(), **namespace)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("mscale", [1.0, 1.17])
@pytest.mark.parametrize("positions", [[0, 1, 2], [1023, 32768, 262143]])
def test_compact_rope_matches_full_table_rows(mscale, positions):
    f = load_source(
        KIMI + "dspark_mla.py", ["_fill_compact_rope_cache"]
    )._fill_compact_rope_cache
    pos = torch.tensor(positions)
    inv = torch.exp(-torch.arange(8).float())
    cache = torch.empty(8, 16)
    actual = f(pos, inv, torch.empty(8, 8), cache, mscale=mscale)
    freqs = pos.float()[:, None] * inv[None, :]
    expected = torch.cat((freqs.cos(), freqs.sin()), dim=-1) * mscale
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.data_ptr() == cache.data_ptr()
    with pytest.raises(ValueError, match="too small"):
        f(torch.arange(9), inv, torch.empty(8, 8), cache, mscale=mscale)


def test_attn_res_workspace_storage_survives_shorter_chunks():
    ns = load_source(
        KIMI + "model.py",
        ["_get_attn_res_workspace", "reserve_attn_res_workspace"],
        parent="KimiLinearModel",
        _release_cuda_cache_before_retained_allocation=lambda _: None,
    )
    model = torch.nn.Module()
    model.register_buffer("_attn_res_workspace", None)
    model.num_attn_res_blocks = 3
    model.use_attn_res = True
    model._max_num_batched_tokens = 16
    model._model_dtype = torch.bfloat16
    model.config = SimpleNamespace(hidden_size=4)
    model.register_parameter(
        "weight", torch.nn.Parameter(torch.empty(1, dtype=torch.bfloat16))
    )
    ns.reserve_attn_res_workspace(model)
    original = model._attn_res_workspace
    active = ns._get_attn_res_workspace(model, torch.empty(3, 4, dtype=torch.bfloat16))
    assert active.shape == (3, 3, 4)
    assert active.data_ptr() == original.data_ptr()
    # Individual block rows must remain contiguous for alias-safe kernels.
    assert active[:, 1].is_contiguous()
    ns.reserve_attn_res_workspace(model)
    assert model._attn_res_workspace is original


@pytest.mark.parametrize("final,write", [(False, False), (False, True), (True, True)])
def test_attn_res_never_overwrites_final_committed_block(final, write):
    outputs = []

    def record(*args, **kwargs):
        output = kwargs["output"]
        outputs.append(output)
        return args[0].clone() if output is None else output

    ns = load_source(
        KIMI + "model.py",
        ["_post_attn_norm"],
        parent="KimiDecoderLayer",
        attn_res=record,
    )
    norm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-5)
    layer = SimpleNamespace(
        use_attn_res=True,
        reuse_attn_res_output=True,
        is_block_write_layer=write,
        is_final_block_write_layer=final,
        prev_valid_blocks=2,
        mlp_res_norm=norm,
        mlp_res_proj=SimpleNamespace(weight=torch.ones(1, 4)),
        post_attention_layernorm=norm,
    )
    hidden, prefix, blocks = torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 3, 4)
    ns._post_attn_norm(layer, hidden, blocks, prefix)
    assert outputs[0] is (None if final else prefix if write else hidden)


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_padded_tp_load_reconstructs_logical_weights(dim, rank):
    f = load_source(KIMI + "model.py", ["_load_padded_tp_shard"])._load_padded_tp_shard
    weight = torch.arange(21).reshape(3, 7) if dim else torch.arange(21).reshape(7, 3)
    shape = list(weight.shape)
    shape[dim] = 2
    param = torch.nn.Parameter(torch.empty(shape), requires_grad=False)
    f(param, weight, dim, rank)
    padded = torch.cat((weight, torch.zeros_like(weight.narrow(dim, 0, 1))), dim=dim)
    torch.testing.assert_close(
        param, padded.narrow(dim, rank * 2, 2), check_dtype=False
    )


def test_merged_qkv_gather_preserves_projection_order():
    f = load_source(
        KIMI + "mla.py", ["_restore_merged_output_order"]
    )._restore_merged_output_order
    logical = torch.arange(24).reshape(2, 12)
    first, second = logical.split([8, 4], dim=-1)
    ranks = torch.cat(
        [
            torch.cat((first[:, r * 2 : r * 2 + 2], second[:, r : r + 1]), dim=-1)
            for r in range(4)
        ],
        dim=-1,
    )
    torch.testing.assert_close(f(ranks, [8, 4], 4), logical)
    with pytest.raises(ValueError, match="divisible"):
        f(ranks, [7, 5], 4)


@pytest.mark.parametrize("size,interleave", [(1, 1), (4, 1), (4, 3), (16, 1)])
def test_dcp_lengths_cover_every_token_once(size, interleave):
    f = load_source(
        "vllm/v1/attention/backends/mla/b12x_mla.py",
        ["_dcp_local_seq_lens_from_global"],
    )._dcp_local_seq_lens_from_global
    global_lengths = torch.tensor([0, 1, 7, 16, 65, 1025], dtype=torch.int32)
    all_outputs = []
    for rank in range(size):
        out, scratch = (
            torch.empty_like(global_lengths),
            torch.empty_like(global_lengths),
        )
        f(
            out,
            scratch,
            global_lengths,
            dcp_size=size,
            dcp_rank=rank,
            interleave=interleave,
        )
        expected = torch.tensor(
            [
                sum(
                    (token // interleave) % size == rank for token in range(int(length))
                )
                for length in global_lengths
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(out, expected)
        all_outputs.append(out)
    torch.testing.assert_close(sum(all_outputs), global_lengths)


def test_b12x_adapter_requests_bindable_scratch_plan(monkeypatch):
    fused_moe = pytest.importorskip("b12x.moe.fused_moe")
    from b12x.moe.fused_moe import _impl

    # CPU planner gate with the Spark SM count supplied explicitly. This does
    # not compile or execute a GPU kernel.
    monkeypatch.setattr(_impl, "get_num_sm", lambda _: 40)
    weight_plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="fp4_e8m0_k32",
        activation="situ",
        params_dtype=torch.bfloat16,
        num_experts=896,
        hidden_size=2048,
        intermediate_size=192,
        w13_layout="w31",
    )
    ns = load_source(
        "vllm/model_executor/layers/fused_moe/b12x.py",
        ["_b12x_moe_execution_plan", "_b12x_scratch_nbytes"],
        _require_b12x_fused_moe=lambda: fused_moe,
    )
    plan = ns._b12x_moe_execution_plan(
        tokens=9,
        topk=16,
        prepared=SimpleNamespace(w1_fp4=torch.empty(0), plan=weight_plan),
        quant_mode="w4a16",
        apply_router_weight_on_input=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
    )
    assert callable(plan.bind)
    assert ns._b12x_scratch_nbytes(plan) > 0
    assert plan.caps.weight_plan == weight_plan
    assert plan.caps.max_tokens == 9


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_b12x_dense_mla_cutlass_kernel_construction(dtype):
    dense_mla = pytest.importorskip("b12x.attention.dense_mla")
    from b12x.attention.dense_mla._forward import DenseMlaForwardKernel
    from b12x.attention.dense_mla._layout import make_smem_layout
    from b12x.attention.dense_mla._merge import DenseMlaMergeKernel

    caps = dense_mla.Caps(
        device="cpu",
        mode="decode",
        kv_dtype=dtype,
        num_q_heads=64,
        page_size=1024,
        max_total_q=9,
        max_batch=9,
        max_cache_tokens=65536,
        max_page_table_width=64,
        num_cache_pages=10000,
        use_cuda_graph=True,
    )
    plan = dense_mla.plan(caps)
    assert plan.shapes_and_dtypes()[0][1] == torch.uint8
    layout = make_smem_layout(
        query_tile=1, fp8=dtype == torch.float8_e4m3fn, qk_dim=576
    )
    kernel = DenseMlaForwardKernel(
        layout=layout,
        page_size=1024,
        num_heads=64,
        num_splits=4,
        chunks_per_split=16,
        query_tile=1,
        fp8=dtype == torch.float8_e4m3fn,
        qk_dim=576,
        value_dim=512,
        window_size=None,
    )
    assert callable(kernel)
    assert callable(DenseMlaMergeKernel(num_splits=4, value_dim=512))
    assert layout.total_bytes <= 99 * 1024
