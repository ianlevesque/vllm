# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage of V4.1's independent SM12x index-key cache groups.

Exercise real topology construction, manager grouping, packed allocation, and
page addressing. Stub only unrelated math modules and CUDA event creation.
The SM12x cases fail on the original source: layer20 exposes 128 index states.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CompilationConfig, set_current_vllm_config
from vllm.model_executor.layers import sparse_attn_indexer
from vllm.models.deepseek_v4_1 import attention
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.indexer import DeepseekV4IndexerBackend
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import KVCacheLayout
from vllm.v1.worker.utils import (
    AttentionGroup,
    allocate_kv_cache,
    prepare_kernel_block_sizes,
)


class _NoMath(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.block_size = kwargs.get("block_size")


class _MLABackend:
    @staticmethod
    def get_supported_kernel_block_sizes():
        return [128]


class _Attention(attention.DeepseekV4Attention):
    backend_cls = _MLABackend
    swa_backend_cls = _MLABackend
    use_fp8_ds_mla_layout = True
    swa_cache_block_size = 64

    @classmethod
    def get_padded_num_q_heads(cls, num_heads):
        return num_heads

    def forward_mqa(self, *args, **kwargs):
        raise AssertionError("This CPU test must not launch attention")

    def _o_proj(self, *args, **kwargs):
        raise AssertionError("This CPU test must not launch a projection")


def _construct_layers(monkeypatch, capability):
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda family, device_id=0: capability // 10 == family // 10,
    )
    monkeypatch.setattr(attention, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(sparse_attn_indexer, "has_deep_gemm", lambda: True)
    for name in (
        "ColumnParallelLinear",
        "MergedColumnParallelLinear",
        "ReplicatedLinear",
        "RowParallelLinear",
        "RMSNorm",
        "DeepseekCompressor",
        "DeepseekV4SWACache",
        "build_deepseek_v4_rope",
    ):
        monkeypatch.setattr(attention, name, _NoMath)
    monkeypatch.setattr(torch.cuda, "Event", lambda: None)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=1024,
            hf_config=SimpleNamespace(
                model_type="deepseek_v41",
                hidden_size=64,
                num_attention_heads=64,
                q_lora_rank=32,
                o_lora_rank=32,
                head_dim=512,
                qk_rope_head_dim=64,
                o_groups=8,
                sliding_window=128,
                compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
                kv_source_layer_ids=[2, 8, 14, 20],
                index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
                candidate_source_layer_id=20,
                candidate_topk_blocks=2048,
                candidate_block_size=8,
                num_hidden_layers=40,
                rms_norm_eps=1e-6,
                max_position_embeddings=1024,
                index_topk=512,
                index_n_heads=32,
                index_head_dim=128,
            ),
        ),
        cache_config=SimpleNamespace(
            block_size=128,
            cache_dtype="fp8",
            num_gpu_blocks_override=16,
            prefix_cache_retention_interval=None,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC,
        ),
        compilation_config=CompilationConfig(custom_ops=["all"]),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=8, disable_hybrid_kv_cache_manager=False
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        attention_config=SimpleNamespace(
            resolve_indexer_kv_dtype=lambda default: "fp8"
        ),
        kernel_config=SimpleNamespace(enable_jit_warmup=False),
        quant_config=None,
        speculative_config=None,
        use_v2_model_runner=True,
    )
    topk = torch.empty(8, 512, dtype=torch.int32)
    candidates = torch.empty(8, 2048, dtype=torch.int32)
    with set_current_vllm_config(config):
        layers = {
            i: _Attention(
                config,
                prefix=f"model.layers.{i}.self_attn",
                topk_indices_buffer=topk,
                candidate_block_buffer=candidates,
            )
            for i in (2, 8, 14, 20, 24, 28, 32, 36)
        }
    return config, layers


@pytest.mark.parametrize("capability", [120, 121])
def test_indexer_groups_and_sharing_on_sm12x(monkeypatch, capability):
    config, layers = _construct_layers(monkeypatch, capability)
    owners = (2, 8, 14, 20)
    index_caches = {i: layers[i].indexer.k_cache for i in owners}
    specs = {}
    backends = {}
    for i in owners:
        layer, index_cache = layers[i], index_caches[i]
        index_spec = index_cache.get_kv_cache_spec(config)
        expected_block_size = 128 if i < 20 else 64
        assert index_spec.block_size == expected_block_size, (
            f"layer{i}: expected indexer block{expected_block_size}, "
            f"got{index_spec.block_size}"
        )
        assert index_spec.num_states == 64
        mla_spec = layer.get_kv_cache_spec(config)
        assert mla_spec.block_size == 128
        assert mla_spec.num_states == (64 if i < 20 else 128)
        for module, spec in ((layer, mla_spec), (index_cache, index_spec)):
            specs[module.prefix] = spec
            backends[module.prefix] = module.get_attn_backend()
    for i in (24, 28, 32, 36):
        assert layers[i].indexer.k_cache is index_caches[20]
        assert layers[i].indexer.indexer_op.k_cache is index_caches[20]
        assert layers[i].get_kv_cache_spec(config) is None
        assert layers[i].compressed_cache_prefix == layers[20].prefix
    assert layers[20].indexer.indexer_op.candidate_write
    assert all(
        not layers[i].indexer.indexer_op.candidate_write for i in (24, 28, 32, 36)
    )

    groups = get_kv_cache_groups(config, specs)
    cache_config = get_kv_cache_config_from_groups(config, groups, available_memory=0)
    attn_groups = [
        [
            AttentionGroup(backends[name], [name], specs[name], gid)
            for name in group.layer_names
        ]
        for gid, group in enumerate(groups)
    ]
    kernel_sizes = prepare_kernel_block_sizes(cache_config, attn_groups)
    for gid, group in enumerate(groups):
        assert kernel_sizes[gid] == group.kv_cache_spec.block_size
        if index_caches[20].prefix in group.layer_names:
            assert group.layer_names == [index_caches[20].prefix]
            assert kernel_sizes[gid] == 64
    views = allocate_kv_cache(
        cache_config, torch.device("cpu"), KVCacheLayout.BLHNC, kernel_sizes
    )
    for i in owners:
        index_cache = index_caches[i]
        index_cache.bind_kv_cache(views[index_cache.prefix])
        assert index_cache.kv_cache.shape == (16, 64, 132)
        assert index_cache.kv_cache.stride(1) == 132
        assert views[layers[i].prefix].shape[2] == (64 if i < 20 else 128)

    # The two groups own disjoint pool blocks. The same original token uses
    # independent tables: CSA1 index keys turn pages at64, MLA latents at128.
    index_table, mla_table = [1, 3, 4], [2, 5]
    index_view = index_caches[20].kv_cache
    mla_view = views[layers[20].prefix]
    positions = (0, 63, 64, 65, 127, 128, 191)
    for tag, pos in enumerate(positions, 1):
        index_view[index_table[pos // 64], pos % 64, 0] = tag
        mla_view[mla_table[pos // 128], 0, pos % 128, 0] = tag + 32
    for tag, pos in enumerate(positions, 1):
        for i in (20, 24, 28, 32, 36):
            cache = layers[i].indexer.indexer_op.k_cache.kv_cache
            assert cache[index_table[pos // 64], pos % 64, 0].item() == tag
        assert mla_view[mla_table[pos // 128], 0, pos % 128, 0].item() == tag + 32


@pytest.mark.parametrize("capability", [90, 100])
def test_indexer_geometry_preserved_off_sm12x(monkeypatch, capability):
    config, layers = _construct_layers(monkeypatch, capability)
    for i in (2, 8, 14, 20):
        cache = layers[i].indexer.k_cache
        spec = cache.get_kv_cache_spec(config)
        assert spec.block_size == 128
        assert spec.num_states == (64 if i < 20 else 128)
        assert cache.get_attn_backend().get_supported_kernel_block_sizes() == (
            DeepseekV4IndexerBackend.get_supported_kernel_block_sizes()
        )
