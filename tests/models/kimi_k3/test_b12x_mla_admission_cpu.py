# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run current CUDA selector/admission code without importing GPU extensions.

Only module loading, device discovery and the dense_mla import are substituted.
The backend, inherited capability defaults and CUDA selected-backend decision
are loaded from source. Kernel execution is qualified separately on hardware.
"""

import ast
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Generic, NamedTuple, TypeVar
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(os.environ.get("VLLM_SOURCE_ROOT") or Path(__file__).resolve().parents[3])
BACKEND = "vllm/v1/attention/backend.py"
COMMON = "vllm/model_executor/layers/attention/mla_attention.py"
B12X = "vllm/v1/attention/backends/mla/b12x_mla.py"


def _load(ns, relative, names, *, parent=None):
    nodes = ast.parse((ROOT / relative).read_text()).body
    if parent:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == parent
        ).body
    selected = [
        n
        for n in nodes
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
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
    exec(compile(ast.fix_missing_locations(module), str(ROOT / relative), "exec"), ns)


def _selector(model_type, dcp, *, local_heads=6, pcp=1):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type=model_type,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
            ),
            get_num_attention_heads=lambda _: local_heads,
            max_model_len=262144,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
            cp_kv_cache_interleave_size=1,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
    )
    ns = dict(
        ABC=ABC,
        abstractmethod=abstractmethod,
        dataclass=dataclass,
        Enum=Enum,
        Generic=Generic,
        T=TypeVar("T"),
        NamedTuple=NamedTuple,
        torch=torch,
        logger=Mock(),
        get_current_vllm_config=lambda: config,
        _load_dense_mla=lambda: object(),
        _K3_KV_LORA_RANK=512,
        _K3_QK_NOPE_HEAD_DIM=128,
        _K3_QK_ROPE_HEAD_DIM=64,
        _K3_V_HEAD_DIM=128,
        _MAX_B12X_QUERY_ROWS=1024,
        _MAX_B12X_CACHE_TOKENS=1048576,
        _B12X_QUERY_HEAD_TILE=8,
    )
    _load(
        ns,
        BACKEND,
        ["AttentionType", "MultipleOf", "AttentionBackend", "AttentionImplBase"],
    )
    _load(ns, COMMON, ["MLACommonBackend"])
    # No builder or implementation instance is constructed. Preserve real
    # AttentionImplBase defaults and all B12X capability declarations.
    ns.update(
        MLACommonImpl=ns["AttentionImplBase"],
        MLACommonMetadataBuilder=ns["AttentionImplBase"],
        B12xMLAMetadata=object,
        AttentionCGSupport=SimpleNamespace(UNIFORM_BATCH=1),
        QueryLenSupport=SimpleNamespace(UNIFORM=1),
    )
    _load(
        ns,
        B12X,
        [
            "_max_dcp_local_cache_tokens",
            "_kernel_query_heads",
            "B12xMLAMetadataBuilder",
            "B12xMLABackend",
            "B12xMLAImpl",
        ],
    )
    _load(ns, "vllm/v1/attention/selector.py", ["AttentionSelectorConfig"])
    backend = ns["B12xMLABackend"]
    ns.update(
        _get_attn_backend_class=lambda _: backend,
        _backend_cls_path=lambda cls: f"b12x_mla.{cls.__name__}",
    )
    _load(
        ns,
        "vllm/platforms/cuda.py",
        ["get_attn_backend_cls"],
        parent="CudaPlatformBase",
    )
    platform = type(
        "SelectorPlatform",
        (),
        {
            "get_device_capability": classmethod(
                lambda cls: SimpleNamespace(major=12, minor=1)
            ),
            "get_attn_backend_cls": ns["get_attn_backend_cls"],
        },
    )
    args = dict(
        head_size=576,
        dtype=torch.bfloat16,
        kv_cache_dtype="fp8",
        block_size=1024,
        use_mla=True,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        attn_type="decoder",
        use_dcp=dcp > 1,
        use_pcp=pcp > 1,
    )
    return platform, ns["AttentionSelectorConfig"](**args), backend


@pytest.mark.parametrize(
    "model_type,dcp,heads", [("kimi_linear", 4, 6), ("kimi_k2", 16, 4)]
)
@pytest.mark.parametrize("kv_dtype", ["bfloat16", "fp8"])
def test_selected_b12x_admits_qualified_dcp(model_type, dcp, heads, kv_dtype):
    platform, config, _ = _selector(model_type, dcp, local_heads=heads)
    config = config._replace(kv_cache_dtype=kv_dtype)
    assert (
        platform.get_attn_backend_cls("B12X_MLA", config) == "b12x_mla.B12xMLABackend"
    )


def test_selector_negative_control_missing_dcp_declaration(monkeypatch):
    platform, config, backend = _selector("kimi_linear", 4)
    monkeypatch.delattr(backend.get_impl_cls(), "supports_dcp")
    with pytest.raises(ValueError, match="DCP not supported"):
        platform.get_attn_backend_cls("B12X_MLA", config)


def test_replicated_noncausal_draft_admitted_without_dcp():
    platform, config, _ = _selector("k3_dspark", 1, local_heads=96)
    assert (
        platform.get_attn_backend_cls("B12X_MLA", config._replace(use_non_causal=True))
        == "b12x_mla.B12xMLABackend"
    )


def test_noncausal_sharded_draft_remains_rejected():
    platform, config, _ = _selector("k3_dspark", 4)
    with pytest.raises(
        ValueError, match="non-causal MLA attention with DCP not supported"
    ):
        platform.get_attn_backend_cls("B12X_MLA", config._replace(use_non_causal=True))


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"use_pcp": True}, "PCP not supported"),
        ({"has_sliding_window": True}, "sliding window not supported"),
        ({"has_sink": True}, "attention sinks not supported"),
        ({"head_size": 512}, "head_size not supported"),
        ({"dtype": torch.float16}, "dtype not supported"),
        ({"kv_cache_dtype": "fp8_ds_mla"}, "kv_cache_dtype not supported"),
    ],
)
def test_unsupported_capabilities_remain_rejected(change, reason):
    platform, config, _ = _selector("kimi_linear", 4)
    with pytest.raises(ValueError, match=reason):
        platform.get_attn_backend_cls("B12X_MLA", config._replace(**change))


def test_nonintegral_dcp_kernel_head_tile_remains_rejected():
    platform, config, _ = _selector("kimi_linear", 2)
    with pytest.raises(ValueError, match="multiple of 8 query heads after DCP"):
        platform.get_attn_backend_cls("B12X_MLA", config)
