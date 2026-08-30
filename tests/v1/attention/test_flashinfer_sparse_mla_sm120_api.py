# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    _required_sm120_sparse_topk,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backends.mla import flashinfer_mla_sparse_sm120 as sm120_module
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
    FlashInferMLASparseSM120Impl,
    _pad_query_heads_for_sm120,
    _reshape_kv_cache_for_sm120_decode,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _fake_vllm_config(model_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type, index_topk=2048),
        ),
    )


def test_sm120_backend_uses_dedicated_backend_name() -> None:
    assert FlashInferMLASparseSM120Backend.get_name() == "FLASHINFER_MLA_SPARSE_SM120"
    assert (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120.get_class()
        is FlashInferMLASparseSM120Backend
    )


def test_sm120_backend_uses_sparse_mqa_for_prefill() -> None:
    impl_cls = FlashInferMLASparseSM120Backend.get_impl_cls()

    assert impl_cls.is_sparse
    assert not impl_cls.supports_dense_mha_prefill


def test_sm120_tp16_query_heads_are_zero_padded_to_kernel_minimum() -> None:
    query = torch.arange(2 * 4 * 7, dtype=torch.float32).view(2, 4, 7)

    padded = _pad_query_heads_for_sm120(query)

    assert padded.shape == (2, 8, 7)
    torch.testing.assert_close(padded[:, :4], query)
    assert torch.count_nonzero(padded[:, 4:]) == 0


def test_sm120_supported_query_head_count_is_unchanged() -> None:
    query = torch.randn(2, 8, 7)

    assert _pad_query_heads_for_sm120(query) is query


def test_sm120_decode_virtually_splits_large_kv_blocks() -> None:
    cache = (
        torch.arange(3 * 128 * 656, dtype=torch.int64).to(torch.uint8).view(3, 128, 656)
    )

    decode_view = _reshape_kv_cache_for_sm120_decode(cache)

    assert decode_view.shape == (6, 64, 656)
    assert (
        decode_view.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    )


def test_sm120_decode_rejects_non_divisible_kv_blocks() -> None:
    cache = torch.empty(2, 96, 656, dtype=torch.uint8)

    with pytest.raises(ValueError, match="divisible by 64"):
        _reshape_kv_cache_for_sm120_decode(cache)


def test_v32_glm_sm120_backend_accepts_glm_block_size(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)

    with set_current_vllm_config(_fake_vllm_config("glm4_moe")):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=256,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_sm120_dsv4_capability_checks_exact_dispatch_shape(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        _DECODE_DSV4_DISPATCH=frozenset({(32, 128), (32, 192)})
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(fi_utils, "_get_submodule", lambda _name: fake_module)
    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()

    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 128)
    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 192)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 256)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(16, 192)

    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()


def test_sm120_dsv4_required_topk_tracks_dspark_width() -> None:
    causal = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=False),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )
    dspark = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=True),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )

    assert _required_sm120_sparse_topk(causal, 128) == 128
    assert _required_sm120_sparse_topk(dspark, 128) == 192


def _make_dcp_impl(
    monkeypatch,
    num_local_heads: int,
    topk_indices_buffer: torch.Tensor,
) -> FlashInferMLASparseSM120Impl:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(
        sm120_module,
        "_get_workspace_buffer",
        lambda device: torch.empty(1, dtype=torch.uint8, device=device),
    )
    with set_current_vllm_config(_fake_vllm_config("glm4_moe")):
        impl = FlashInferMLASparseSM120Impl(
            num_heads=num_local_heads,
            head_size=576,
            scale=1.0,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8_ds_mla",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            indexer=SimpleNamespace(topk_indices_buffer=topk_indices_buffer),
            kv_lora_rank=512,
            qk_nope_head_dim=512,
            qk_rope_head_dim=64,
        )
    impl.dcp_world_size = 2
    impl.dcp_rank = 1
    impl.need_to_return_lse_for_decode = True
    return impl


def _metadata(num_tokens: int, topk_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        req_id_per_token=torch.zeros(num_tokens, dtype=torch.int32),
        block_table=torch.arange(8, dtype=torch.int32).reshape(2, 4),
        block_size=64,
        cp_kv_cache_interleave_size=1,
        topk_tokens=topk_tokens,
    )


def test_sm120_sparse_mla_dcp_fallback_plumbs_lse_and_valid_counts(
    monkeypatch,
) -> None:
    """>64-token batches take the trtllm-gen wrapper; DCP filtering, seq_lens,
    return_lse, LSE normalization and empty-row masking must be plumbed
    (adapted from upstream PR #47779)."""
    num_tokens = 65  # > _SM120_DECODE_MAX_TOKENS: routes to the wrapper path
    topk_tokens = 4
    gathered_heads = 8  # TP16 x DCP2: 4 local heads all-gathered to 8
    kv_lora_rank = 512

    topk_indices = torch.zeros(num_tokens, topk_tokens, dtype=torch.int32)
    impl = _make_dcp_impl(
        monkeypatch, num_local_heads=4, topk_indices_buffer=topk_indices
    )

    topk_indices_physical = torch.zeros(num_tokens, topk_tokens, dtype=torch.int32)
    topk_indices_physical[1] = -1
    seq_lens = torch.full((num_tokens,), topk_tokens, dtype=torch.int32)
    seq_lens[1] = 0
    captured: dict[str, dict[str, object]] = {}

    def fake_filter_and_convert_dcp_index(*args: object, **kwargs: object):
        captured["dcp_kwargs"] = kwargs
        return topk_indices_physical, seq_lens

    def fake_flashinfer_decode(**kwargs: object):
        out = kwargs["out"]
        assert isinstance(out, torch.Tensor)
        assert out.shape == (num_tokens, 1, gathered_heads, kv_lora_rank)
        assert kwargs["seq_lens"] is seq_lens
        assert kwargs["return_lse"] is True
        assert isinstance(kwargs["block_tables"], torch.Tensor)
        assert kwargs["block_tables"].shape == (num_tokens, 1, topk_tokens)
        out.fill_(1.0)
        lse = torch.ones(num_tokens, 1, gathered_heads, dtype=torch.float32)
        return out, lse

    monkeypatch.setattr(
        sm120_module,
        "triton_filter_and_convert_dcp_index",
        fake_filter_and_convert_dcp_index,
    )
    monkeypatch.setattr(
        fi_utils,
        "flashinfer_trtllm_batch_decode_with_kv_cache_mla",
        fake_flashinfer_decode,
    )

    q = torch.zeros(num_tokens, gathered_heads, 576, dtype=torch.bfloat16)
    kv_cache = torch.zeros(1, 64, 656, dtype=torch.uint8)

    out, lse = impl.forward_mqa(
        q, kv_cache, _metadata(num_tokens, topk_tokens), SimpleNamespace()
    )

    assert out.shape == (num_tokens, gathered_heads, kv_lora_rank)
    assert lse is not None
    assert lse.shape == (num_tokens, gathered_heads)
    assert torch.all(out[0] == 1)
    assert torch.all(out[1] == 0)
    assert torch.isneginf(lse[1]).all()
    assert torch.all(lse[2] == 1.0)
    assert captured["dcp_kwargs"]["dcp_size"] == 2
    assert captured["dcp_kwargs"]["dcp_rank"] == 1
    assert captured["dcp_kwargs"]["return_valid_counts"] is True


def test_sm120_sparse_mla_dcp_fast_decode_returns_kernel_lse(
    monkeypatch,
) -> None:
    """<=64-token decode takes the direct sparse_mla_sm120_decode_dsv3_2 path;
    under DCP it must hand back the base-2 LSE the kernel fills and mask rows
    whose top-k slots all live on other ranks."""
    import sys
    from types import ModuleType

    num_tokens = 3
    topk_tokens = 2048  # fast path requires effective topk == 2048
    gathered_heads = 8
    kv_lora_rank = 512

    topk_indices = torch.zeros(num_tokens, topk_tokens, dtype=torch.int32)
    impl = _make_dcp_impl(
        monkeypatch, num_local_heads=4, topk_indices_buffer=topk_indices
    )

    topk_indices_physical = torch.zeros(num_tokens, topk_tokens, dtype=torch.int32)
    topk_indices_physical[1] = -1
    seq_lens = torch.tensor([5, 0, 7], dtype=torch.int32)

    monkeypatch.setattr(
        sm120_module,
        "triton_filter_and_convert_dcp_index",
        lambda *args, **kwargs: (topk_indices_physical, seq_lens),
    )

    def fake_decode_workspace(_workspace, **kwargs):
        return (
            torch.empty(1, dtype=torch.bfloat16),
            torch.empty(1, dtype=torch.float32),
        )

    def fake_sparse_decode(**kwargs):
        assert kwargs["indices"] is topk_indices_physical
        kwargs["output"].fill_(1.0)
        kwargs["out_lse"].fill_(3.0)
        return kwargs["output"]

    fi_pkg = ModuleType("flashinfer")
    fi_mla = ModuleType("flashinfer.mla")
    fi_core = ModuleType("flashinfer.mla._core")
    fi_core._sparse_mla_decode_workspace = fake_decode_workspace
    fi_sm120 = ModuleType("flashinfer.mla._sparse_mla_sm120")
    fi_sm120._MODEL_TYPE_GLM_NSA = 2
    fi_sm120.sparse_mla_sm120_decode_dsv3_2 = fake_sparse_decode
    fi_pkg.mla = fi_mla
    fi_mla._core = fi_core
    fi_mla._sparse_mla_sm120 = fi_sm120
    for name, mod in (
        ("flashinfer", fi_pkg),
        ("flashinfer.mla", fi_mla),
        ("flashinfer.mla._core", fi_core),
        ("flashinfer.mla._sparse_mla_sm120", fi_sm120),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    q = torch.zeros(num_tokens, gathered_heads, 576, dtype=torch.bfloat16)
    kv_cache = torch.zeros(1, 64, 656, dtype=torch.uint8)

    out, lse = impl.forward_mqa(
        q, kv_cache, _metadata(num_tokens, topk_tokens), SimpleNamespace()
    )

    assert out.shape == (num_tokens, gathered_heads, kv_lora_rank)
    assert lse is not None
    assert lse.shape == (num_tokens, gathered_heads)
    assert torch.all(out[0] == 1)
    assert torch.all(out[1] == 0)
    assert torch.all(lse[0] == 3.0)
    assert torch.isneginf(lse[1]).all()
    assert torch.all(lse[2] == 3.0)


def test_sm120_sparse_mla_declares_base2_decode_lse() -> None:
    assert FlashInferMLASparseSM120Impl.can_return_lse_for_decode is True
    assert FlashInferMLASparseSM120Impl.lse_base_on_e is False
