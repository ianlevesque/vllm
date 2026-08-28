# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
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
    cache = torch.arange(3 * 128 * 656, dtype=torch.int64).to(torch.uint8).view(
        3, 128, 656
    )

    decode_view = _reshape_kv_cache_for_sm120_decode(cache)

    assert decode_view.shape == (6, 64, 656)
    assert decode_view.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()


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
