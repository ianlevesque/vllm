# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU behavior gates for the shared Spark source.

Run with pytest --confcutdir=tests/standalone
  tests/standalone/test_shared_source_carries.py.
These execute the real functions selected from the source AST, with CUDA launch
and distributed plumbing injected at their boundaries. Tensor ownership,
checkpoint mapping, cache lifecycle, and FI call contracts are real CPU tests;
GPU numerical/graph and full-model qualification remain separate gates.
"""

from __future__ import annotations

import ast
import dataclasses
import math  # noqa: F401 - used by AST-loaded rotary helpers
import os
import re
import sys
from collections import OrderedDict  # noqa: F401 - used by loaded cache manager
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(os.environ.get("VLLM_CARRY_TEST_ROOT", Path(__file__).resolve().parents[2]))


def load(relative, path, source_text=None, **namespace):
    tree = ast.parse(
        source_text if source_text is not None else (ROOT / relative).read_text()
    )
    for name in path.split("."):
        tree = next(n for n in tree.body if getattr(n, "name", None) == name)
    # Decorator effects belong to the caller when loading a single method.
    if isinstance(tree, (ast.FunctionDef, ast.AsyncFunctionDef)):
        tree.decorator_list = []
    scope = dict(globals(), **namespace)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            tree,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(ROOT / relative), "exec"), scope
    )
    return scope[path.split(".")[-1]]


MIMO = "vllm/model_executor/models/mimo_v2.py"
M3 = "vllm/models/minimax_m3/nvidia/model.py"
PARSER = "vllm/reasoning/minimax_m3_reasoning_parser.py"
SM120 = "vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"


@pytest.mark.parametrize("first", ["weight", "weight_scale_inv"])
def test_mimo_held_half_owns_storage_before_pair_completes(first):
    captured = {}

    def shard(weight, scale, **kwargs):
        captured.update(weight=weight.clone(), weight_scale_inv=scale.clone())
        return weight, scale

    method = load(
        MIMO,
        "MiMoV2Model._try_load_fp8_qkv_proj",
        is_pp_missing_parameter=lambda *_: False,
        _shard_fp8_qkv_proj=shard,
        default_weight_loader=lambda param, value: param.copy_(value),
    )
    prefix = "layers.0.self_attn.qkv_proj"
    tensors = {
        "weight": torch.full((4, 4), 2.0, dtype=torch.float8_e4m3fn),
        "weight_scale_inv": torch.full((4, 4), 3.0),
    }
    params = {
        prefix + "." + name: torch.zeros_like(value) for name, value in tensors.items()
    }
    owner = NS(
        config=NS(num_key_value_heads=1),
        get_submodule=lambda _: NS(
            total_num_heads=1, total_num_kv_heads=1, head_dim=4, v_head_dim=4
        ),
    )
    pending, loaded = {}, set()
    held = tensors[first]
    expected = {key: value.clone() for key, value in tensors.items()}
    assert method(owner, prefix + "." + first, held, pending, params, loaded, 0, 1)
    held.fill_(9.0)  # The real loader is permitted to reuse its staging buffer.
    second = "weight_scale_inv" if first == "weight" else "weight"
    assert method(
        owner, prefix + "." + second, tensors[second], pending, params, loaded, 0, 1
    )
    assert not pending
    assert loaded == set(params)
    for key, value in expected.items():
        torch.testing.assert_close(captured[key].float(), value.float(), rtol=0, atol=0)
        torch.testing.assert_close(
            params[prefix + "." + key].float(), value.float(), rtol=0, atol=0
        )


@pytest.mark.parametrize("first", ["weight", "weight_scale_inv"])
def test_mimo_mtp_held_half_owns_storage_across_load_calls(first):
    source = ROOT / "vllm/model_executor/models/mimo_v2_mtp.py"
    tree = ast.parse(source.read_text())
    owner_class = next(
        c.name
        for c in tree.body
        if isinstance(c, ast.ClassDef)
        and any(
            isinstance(m, ast.FunctionDef)
            and m.name == "load_weights"
            and "_pending_qkv_proj" in ast.unparse(m)
            for m in c.body
        )
    )
    prefix = "model.mtp.layers.0.self_attn.qkv_proj"
    tensors = {
        "weight": torch.full((4, 4), 2.0, dtype=torch.float8_e4m3fn),
        "weight_scale_inv": torch.full((4, 4), 3.0),
    }
    params = {
        prefix + "." + key: torch.zeros_like(value) for key, value in tensors.items()
    }
    method = load(
        str(source.relative_to(ROOT)),
        owner_class + ".load_weights",
        get_tensor_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_world_size=lambda: 1,
        _shard_fp8_qkv_proj=lambda w, s, **_: (w, s),
        default_weight_loader=lambda p, v: p.copy_(v),
    )
    owner = NS(
        config=NS(num_key_value_heads=1),
        named_parameters=lambda: params.items(),
        get_submodule=lambda _: NS(
            total_num_heads=1, total_num_kv_heads=1, head_dim=4, v_head_dim=4
        ),
        get_expert_mapping=lambda: [],
    )
    expected = {key: value.clone() for key, value in tensors.items()}
    method(owner, [(prefix + "." + first, tensors[first])])
    tensors[first].fill_(9.0)
    second = "weight_scale_inv" if first == "weight" else "weight"
    method(owner, [(prefix + "." + second, tensors[second])])
    assert owner._pending_qkv_proj == {}
    for key, value in expected.items():
        torch.testing.assert_close(
            params[prefix + "." + key].float(), value.float(), rtol=0, atol=0
        )


def m3_mapper():
    mapper = load(
        "vllm/model_executor/models/utils.py",
        "WeightsMapper",
        dataclass=dataclasses.dataclass,
        field=dataclasses.field,
    )
    cls = next(
        c
        for c in ast.parse((ROOT / M3).read_text()).body
        if isinstance(c, ast.ClassDef) and c.name == "MiniMaxM3SparseForCausalLM"
    )
    stmt = next(
        n
        for n in cls.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "hf_to_vllm_mapper" for t in n.targets
        )
    )
    scope = {"WeightsMapper": mapper, "re": re}
    exec(
        compile(ast.Module(body=[stmt], type_ignores=[]), str(ROOT / M3), "exec"), scope
    )
    return scope["hf_to_vllm_mapper"]


@pytest.mark.parametrize(
    "prefix", ["model.", "model.language_model.", "language_model."]
)
@pytest.mark.parametrize(
    "part,wanted", [("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")]
)
def test_m3_transformers_expert_mapping_and_native_names(prefix, part, wanted):
    mapper = m3_mapper()
    expected_prefix = "" if prefix == "language_model." else "model."
    original = prefix + "layers.5.mlp.experts.17." + part + ".weight"
    expected = (
        expected_prefix + "layers.5.block_sparse_moe.experts.17." + wanted + ".weight"
    )
    assert mapper.map_name(original) == expected
    native = "model.layers.5.block_sparse_moe.experts.17.w1.weight"
    assert mapper.map_name(native) == native
    for dense in [
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
    ]:
        assert mapper.map_name(dense) == dense


@pytest.mark.parametrize(
    "suffix,expected",
    [
        (".mlp.gate.weight", ".block_sparse_moe.gate.weight"),
        (
            ".mlp.shared_experts.down_proj.weight",
            ".block_sparse_moe.shared_experts.down_proj.weight",
        ),
        (".self_attn.indexer.q_proj.weight", ".self_attn.index_q_proj.weight"),
        (".self_attn.indexer.k_norm.weight", ".self_attn.index_k_norm.weight"),
    ],
)
def test_m3_gate_shared_expert_and_indexer_mapping(suffix, expected):
    assert (
        m3_mapper().map_name("model.layers.5" + suffix) == "model.layers.5" + expected
    )


@pytest.mark.parametrize(
    "current,ended", [([], False), ([2], False), ([2, 9, 3], True), ([3], True)]
)
def test_m3_reasoning_only_reads_current_turn(current, ended):
    find = load(PARSER, "MiniMaxM3ReasoningParser._rfind_token_sequence")
    method = load(PARSER, "MiniMaxM3ReasoningParser.is_reasoning_end")
    owner = NS(
        _start_token_ids=[2],
        _end_token_ids=[3],
        _turn_start_token_ids=[1],
        _rfind_token_sequence=find,
    )
    # Literal end markers in system/past assistant messages cannot close this turn.
    assert method(owner, [1, 7, 3, 1, 2, 8, 3, 1] + current) is ended


@dataclasses.dataclass
class LoadConfig:
    load_format: str


@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize(
    "kind", ["instanttensor", "fastsafetensors", "auto", "safetensors"]
)
def test_pp_draft_loader_avoids_world_collective_without_mutating_target(world, kind):
    method = load(
        "vllm/v1/worker/gpu/spec_decode/utils.py",
        "get_pp_safe_draft_load_config",
        get_pp_group=lambda: NS(world_size=world),
        replace=dataclasses.replace,
        logger=Mock(),
    )
    original = LoadConfig(kind)
    selected = method(original)
    fallback = world > 1 and kind in ("instanttensor", "fastsafetensors")
    assert selected.load_format == ("auto" if fallback else kind)
    assert original.load_format == kind
    assert (selected is original) is not fallback


def request(name="a", hashes=("img",)):
    return NS(
        request_id=name,
        mm_features=[
            NS(identifier=h, mm_position=NS(offset=50 + 200 * i, length=100))
            for i, h in enumerate(hashes)
        ],
        get_num_encoder_embeds=lambda _: 100,
        num_computed_tokens=0,
        num_output_placeholders=0,
    )


def encoder_manager():
    return load("vllm/v1/core/encoder_cache_manager.py", "EncoderCacheManager")(200)


def test_upstream_encoder_preemption_and_sequential_hash_reuse():
    manager = encoder_manager()
    first = request()
    manager.allocate(first, 0)
    manager.free(first)
    assert manager.get_freed_mm_hashes() == []
    first.num_computed_tokens = 0
    assert manager.check_and_update_cache(first, 0)
    manager.free(first)
    second = request("b")
    assert manager.check_and_update_cache(second, 0)
    assert manager.get_freed_mm_hashes() == []


def test_upstream_encoder_repeated_positions_pin_until_last_read():
    manager = encoder_manager()
    req = request(hashes=("img", "img"))
    manager.allocate(req, 0)
    manager.check_and_update_cache(req, 1)
    manager.free_encoder_input(req, 0)
    assert manager.cached["img"] == {"a"}
    assert "img" not in manager.freeable
    manager.free_encoder_input(req, 1)
    assert "img" in manager.freeable
    assert manager.get_freed_mm_hashes() == []


def test_upstream_encoder_eviction_requires_recompute_on_resume():
    manager = encoder_manager()
    req = request()
    other = request("b", ("other",))
    manager.allocate(req, 0)
    manager.free(req)
    manager.num_free_slots = 0
    assert manager.can_allocate(other, 0, 1000, 0)
    assert manager.get_freed_mm_hashes() == ["img"]
    assert not manager.check_and_update_cache(req, 0)


def test_upstream_encoder_same_step_reallocation_cancels_stale_free():
    manager = encoder_manager()
    req = request()
    other = request("b", ("other",))
    manager.allocate(req, 0)
    manager.free(req)
    manager.num_free_slots = 0
    assert manager.can_allocate(other, 0, 1000, 0)
    manager.allocate(req, 0)
    assert manager.get_freed_mm_hashes() == []


@pytest.mark.parametrize("lookahead", [0, 1, 7])
def test_upstream_encoder_confirmed_progress_and_draft_margin(lookahead):
    free = load("vllm/v1/core/sched/scheduler.py", "Scheduler._free_encoder_inputs")
    manager = encoder_manager()
    req = request()
    manager.allocate(req, 0)
    owner = NS(
        encoder_cache_manager=manager,
        num_prefill_lookahead=lookahead,
        is_encoder_decoder=False,
        _free_encoder_input=manager.free_encoder_input,
    )
    req.num_output_placeholders = 4
    req.num_computed_tokens = 153 + lookahead
    free(owner, req)
    assert manager.get_cached_input_ids(req) == {0}
    req.num_computed_tokens = 154 + lookahead
    free(owner, req)
    assert manager.get_cached_input_ids(req) == set()


@pytest.mark.parametrize("heads", [4, 8, 16])
@pytest.mark.parametrize("tokens", [1, 3, 64, 65])
@pytest.mark.parametrize("need_lse", [False, True])
def test_sm120_kernel_contract_preserves_heads_and_zeroes_empty_dcp_rows(
    monkeypatch, heads, tokens, need_lse
):
    calls = []

    def kernel(**kw):
        calls.append(kw)
        output = kw["out"]
        output.fill_(5.0)
        lse = torch.full((tokens, 1, max(8, heads)), 7.0)
        if need_lse:
            output[0].fill_(float("nan"))
            lse[0].fill_(float("nan"))
            return output, lse
        return output

    monkeypatch.setitem(
        sys.modules,
        "vllm.utils.flashinfer",
        NS(flashinfer_trtllm_batch_decode_with_kv_cache_mla=kernel),
    )
    method = load(SM120, "FlashInferMLASparseSM120Impl._run_mqa_kernel")
    owner = NS(
        kv_lora_rank=512,
        _workspace_buffer=torch.zeros(1, dtype=torch.uint8),
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        scale=0.5,
        kv_scale_format="arbitrary_fp32",
        need_to_return_lse_for_decode=need_lse,
    )
    query = torch.randn(tokens, heads, 512, dtype=torch.bfloat16)
    lengths = torch.full((tokens,), 2, dtype=torch.int32) if need_lse else None
    if lengths is not None:
        lengths[0] = 0
    output, lse = method(
        owner,
        query,
        torch.zeros(2, 64, 656, dtype=torch.uint8),
        torch.zeros(tokens, 2048, dtype=torch.int32),
        lengths,
    )
    assert output.shape == (tokens, heads, 512)
    assert calls[0]["qk_rope_head_dim"] == 0  # native NoPE, no obsolete fake RoPE
    assert calls[0]["query"].shape == (tokens, 1, max(8, heads), 512)
    if heads < 8:
        assert torch.count_nonzero(calls[0]["query"][:, :, heads:]) == 0
    if need_lse:
        assert lse.shape == (tokens, heads)
        assert not torch.isnan(output).any()
        assert (output[0] == 0).all() and torch.isneginf(lse[0]).all()
        assert (output[1:] == 5).all() and (lse[1:] == 7).all()
    else:
        assert lse is None and (output == 5).all()


@pytest.mark.parametrize(
    "index_topk,accepted", [(2044, True), (2048, True), (0, False), (-1, False)]
)
def test_sm120_backend_accepts_glm_effective_sparse_width(
    monkeypatch, index_topk, accepted
):
    monkeypatch.setitem(
        sys.modules,
        "vllm.config",
        NS(
            get_current_vllm_config=lambda: NS(
                model_config=NS(hf_text_config=NS(index_topk=index_topk, index_kpool=4))
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils.flashinfer",
        NS(has_flashinfer_sparse_mla_sm120=lambda: True),
    )
    method = load(
        "vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py",
        "FlashInferMLASparseSM120Backend.supports_combination",
    )
    error = method(
        None,
        512,
        torch.bfloat16,
        "fp8_ds_mla",
        128,
        True,
        False,
        True,
        False,
        NS(major=12),
    )
    assert (error is None) is accepted


@pytest.mark.parametrize("dcp", [1, 2])
@pytest.mark.parametrize("block_stride", [128, 192])
def test_sm120_forward_passes_physical_stride_and_shard_lengths(dcp, block_stride):
    calls = []
    physical = torch.tensor([[0, 1, -1], [128, -1, -1]], dtype=torch.int32)
    lengths = torch.tensor([2, 1], dtype=torch.int32)

    def convert(*args, **kw):
        calls.append(kw)
        return (physical, lengths) if kw.get("return_valid_counts") else physical

    def run(q, cache, indices, seq_lens):
        assert cache.shape == (6, 64, 656)
        assert indices is physical
        assert (seq_lens is lengths) is (dcp > 1)
        return q, seq_lens

    method = load(
        SM120,
        "FlashInferMLASparseSM120Impl.forward_mqa",
        cast=lambda _, value: value,
        HiSparseMLAIndexGroup=type("HiSparseDummy", (), {}),
        flat_kv_row_view=lambda *_: (
            torch.zeros(384, 656, dtype=torch.uint8),
            block_stride,
        ),
        triton_filter_and_convert_dcp_index=convert,
        triton_convert_req_index_to_global_index=convert,
    )
    owner = NS(
        topk_indices_buffer=torch.zeros(2, 3, dtype=torch.int32),
        index_group=None,
        dcp_world_size=dcp,
        dcp_rank=1 if dcp > 1 else 0,
        _run_mqa_kernel=run,
    )
    metadata = NS(
        req_id_per_token=torch.tensor([0, 1]),
        block_table=torch.tensor([[0], [1]]),
        block_size=128,
        cp_kv_cache_interleave_size=1,
    )
    q = torch.randn(2, 8 if dcp > 1 else 4, 512)
    out, lse = method(owner, q, torch.empty(1), metadata, None)
    assert out is q
    assert calls[0]["BLOCK_STRIDE_ROWS"] == block_stride
    assert calls[0]["BLOCK_SIZE"] == 128
    if dcp > 1:
        assert calls[0]["dcp_size"] == 2 and calls[0]["dcp_rank"] == 1
        assert calls[0]["return_valid_counts"]


def rotary_classes():
    class CustomOp(torch.nn.Module):
        @staticmethod
        def register(_):
            return lambda cls: cls

        def enabled(self):
            return False

    base_path = "vllm/model_executor/layers/rotary_embedding/base.py"
    common_path = "vllm/model_executor/layers/rotary_embedding/common.py"
    scaling_path = (
        "vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py"
    )
    base = load(
        base_path,
        "RotaryEmbeddingBase",
        CustomOp=CustomOp,
        ApplyRotaryEmb=lambda **_: None,
    )
    correction_dim = load(common_path, "yarn_find_correction_dim")
    correction_range = load(
        common_path,
        "yarn_find_correction_range",
        yarn_find_correction_dim=correction_dim,
    )
    ramp = load(common_path, "yarn_linear_ramp_mask")
    scaling = load(
        scaling_path,
        "DeepseekScalingRotaryEmbedding",
        RotaryEmbeddingBase=base,
        yarn_get_mscale=load(scaling_path, "yarn_get_mscale"),
        yarn_find_correction_range=correction_range,
        yarn_linear_ramp_mask=ramp,
    )
    v4 = load(
        scaling_path,
        "DeepseekV4ScalingRotaryEmbedding",
        DeepseekScalingRotaryEmbedding=scaling,
    )
    return base, v4


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("nested", [False, True])
def test_dsv4_ratio1_rope_cache_stays_fp32_and_unscaled(dtype, nested):
    base, v4 = rotary_classes()
    get_rope = load(
        "vllm/model_executor/layers/rotary_embedding/__init__.py",
        "get_rope",
        RotaryEmbedding=base,
        DeepseekV4ScalingRotaryEmbedding=v4,
        DeepseekScalingRotaryEmbedding=v4,
        _ROPE_DICT={},
    )
    builder = load(
        "vllm/models/deepseek_v4/common/rope.py",
        "build_deepseek_v4_rope",
        get_rope=get_rope,
    )
    params = dict(
        rope_type="yarn",
        factor=8.0,
        original_max_position_embeddings=1024,
        beta_fast=32,
        beta_slow=1,
    )
    config = NS(
        rope_theta=10000.0,
        compress_rope_theta=1000000.0,
        rope_parameters={"main": dict(params), "compress": dict(params)}
        if nested
        else dict(params),
    )
    before = repr(config.rope_parameters)
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        rope = builder(
            config,
            head_dim=64,
            rope_head_dim=64,
            max_position_embeddings=8192,
            compress_ratio=1,
        )
    finally:
        torch.set_default_dtype(old_dtype)
    plain = base(64, 64, 8192, 10000.0, False, torch.float32)
    assert rope.scaling_factor == 1.0
    assert rope.cos_sin_cache.dtype == torch.float32
    torch.testing.assert_close(
        rope.cos_sin_cache, plain.cos_sin_cache, atol=1e-3, rtol=1e-3
    )
    assert repr(config.rope_parameters) == before


@pytest.mark.parametrize("smem,expected", [(101376, 1), (102400, 2), (227328, 2)])
def test_grouped_mla_launch_respects_actual_shared_memory(monkeypatch, smem, expected):
    launched = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kw: launched.append(kw)

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: NS(shared_memory_per_block_optin=smem),
    )
    method = load(
        "vllm/v1/attention/ops/triton_decode_attention.py",
        "_decode_grouped_att_m_fwd",
        is_hip_=False,
        triton=NS(
            next_power_of_2=lambda x: 1 << (x - 1).bit_length(),
            cdiv=lambda a, b: (a + b - 1) // b,
        ),
        _fwd_grouped_kernel_stage1=Kernel(),
        _page_stride=lambda *_: 128,
    )
    method(
        torch.zeros(2, 8, 576),
        torch.zeros(2, 64, 1, 576),
        torch.zeros(2, 64, 1, 512),
        torch.zeros(2, 8, 2, 512),
        torch.zeros(2, 2, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        2,
        0.5,
        64,
        0.0,
        1.0,
        1.0,
        is_mla=True,
    )
    assert launched[0]["BLOCK_DMODEL"] == 512
    assert launched[0]["num_stages"] == expected
