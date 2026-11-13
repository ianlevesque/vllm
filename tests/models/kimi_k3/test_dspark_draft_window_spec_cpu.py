# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The K3 DSpark draft keeps its retained-window KV spec.

`K3DSparkAttention` wraps the draft layer's spec in `SelfAddressedMLASpec`.
With `VLLM_DSPARK_DRAFT_KV_WINDOW` the layer returns a
`SlidingWindowMLASpec`, which that full-attention spec cannot be built from
(Kimi K3.1 Pro serves with a 32,768-token draft window). CPU only.
"""

import pytest
import torch

from vllm.models.kimi_k3.common.dsa_cache import SelfAddressedMLASpec
from vllm.models.kimi_k3.nvidia.dspark_mla import draft_kv_cache_spec
from vllm.v1.kv_cache_interface import (
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_sliding_window,
    get_kv_quant_mode,
)

# What MultiHeadLatentAttention.get_kv_cache_spec builds for a replicated
# fp8 draft layer at --block-size 1024.
_KWARGS = dict(
    block_size=1024,
    num_kv_heads=1,
    max_tp_shards=1,
    dcp_sharded=False,
    head_size=576,
    dtype=torch.uint8,
    cache_dtype_str="fp8",
    kv_quant_mode=get_kv_quant_mode("fp8"),
    state_content_bytes=None,
    non_causal_multi_token_decode=True,
)


def test_full_length_draft_cache_is_self_addressed():
    spec = draft_kv_cache_spec(MLAAttentionSpec(**_KWARGS))
    assert isinstance(spec, SelfAddressedMLASpec)
    assert not spec.uses_slot_mapping
    assert spec.non_causal_multi_token_decode


def test_windowed_draft_cache_keeps_its_window():
    windowed = SlidingWindowMLASpec(**_KWARGS, sliding_window=32768)
    # The wrapper has no field for the window's retention state.
    with pytest.raises(TypeError, match="extra_retained_tokens"):
        SelfAddressedMLASpec.from_spec(windowed)

    spec = draft_kv_cache_spec(windowed)
    assert spec is windowed
    assert get_kv_cache_spec_sliding_window(spec) == 32768
    assert not spec.dcp_sharded
    assert spec.non_causal_multi_token_decode


def test_windowed_draft_layers_form_one_uniform_group():
    """The five draft layers group as a windowed bucket of their own; the
    metadata builder reads the window through the group wrapper."""
    layers = {
        f"draft.{i}": draft_kv_cache_spec(
            SlidingWindowMLASpec(**_KWARGS, sliding_window=32768)
        )
        for i in range(5)
    }
    group = UniformTypeKVCacheSpecs.from_specs(layers)
    assert group is not None
    assert isinstance(group.first_spec, SlidingWindowMLASpec)
    assert group.first_spec.sliding_window == 32768
    assert not group.dcp_sharded
    # A target MLA layer is not uniform with the windowed draft bucket.
    target = MLAAttentionSpec(**(_KWARGS | {"dcp_sharded": True, "head_size": 656}))
    assert not UniformTypeKVCacheSpecs.is_uniform_type({**layers, "target.0": target})
