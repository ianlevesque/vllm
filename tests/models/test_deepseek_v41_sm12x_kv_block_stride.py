# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 on SM12x: the packed KV block must stay one group page wide.

The spec set is produced by the model's own ``get_kv_cache_spec`` methods and
the real backends' ``customize_attention_spec`` (no model weights, no GPU), for
the DeepSeek-V4.1-Flash topology served on GB10: FLASHINFER_MLA_SPARSE_DSV41,
fp8_ds_mla, --block-size 128, DSpark k=3, max_model_len 1048576.

Reference numbers are the ones measured on the 128-token-geometry image
(vLLM d124a2f21): an 8/12/20 GiB pool holds 1,570,642 / 2,355,983 / 3,926,665
tokens. With the SM12x FlashInfer backend declaring only 64-token kernel
blocks, the 213,120-byte block is padded to lcm(19008, 37440) = 1,235,520
bytes and a single 1M-token request needs 30.96 GiB.

Run: VLLM_TARGET_DEVICE=cpu PYTHONPATH=<tree> pytest -q test_candidate.py
"""

import os
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest
import torch

import vllm.v1.core.kv_cache_planning as planning
from vllm.config import ModelConfig, VllmConfig
from vllm.models.deepseek_v41 import attention as dsv41_attention
from vllm.models.deepseek_v41.compressor import CompressorBackend, CompressorStateCache
from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
    DeepseekSparseSWAFlashInferBackend,
    DeepseekV4FlashInferSM120Attention,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout
from vllm.v1.worker.utils import (
    AttentionGroup,
    allocate_kv_cache,
    customize_attention_spec,
    prepare_kernel_block_sizes,
    select_common_block_size,
)

GiB = 1 << 30
MAX_MODEL_LEN = 1048576

# deepseek-ai/DeepSeek-V4.1-Flash config.json (text_config), KV-relevant keys.
HF = SimpleNamespace(
    num_hidden_layers=40,
    head_dim=512,
    sliding_window=128,
    compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
    kv_source_layer_ids=[2, 8, 14, 20],
    index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
    index_head_dim=128,
    num_nextn_predict_layers=3,  # DSpark draft layers 40..42 (pure SWA)
)
NUM_SPECULATIVE_TOKENS = 3

# One 128-token block of the widest packed group: 3 x CSA2 latent pages
# (64 x 584 B -> 37,440), 3 x CSA2 index pages (64 x 132 B -> 8,640) and the
# CSA1 latent page (128 x 584 B -> 74,880).
GROUP_PAGE_BYTES = 3 * 37440 + 3 * 8640 + 74880
assert GROUP_PAGE_BYTES == 213120


class _DSparkConfig(SimpleNamespace):
    """What KV planning reads from SpeculativeConfig(method="dspark")."""

    def use_eagle(self):
        return True

    def use_eagle_block_drop(self):
        return False


@pytest.fixture
def sm12x(monkeypatch):
    monkeypatch.setattr(
        type(current_platform),
        "is_device_capability_family",
        classmethod(lambda cls, capability, device_id=0: capability // 10 == 12),
    )


def _vllm_config() -> VllmConfig:
    config = VllmConfig(model_config=ModelConfig(max_model_len=4096))
    config.model_config.max_model_len = MAX_MODEL_LEN
    config.model_config.original_max_model_len = MAX_MODEL_LEN
    config.cache_config.block_size = 128
    config.cache_config.cache_dtype = "fp8_ds_mla"  # what --kv-cache-dtype fp8 resolves to
    config.cache_config.kv_cache_layout = "BLHNC"
    config.scheduler_config.max_num_seqs = 8
    config.scheduler_config.max_num_batched_tokens = 8192
    config.scheduler_config.async_scheduling = True  # two in-flight batches
    config.parallel_config.tensor_parallel_size = 4
    object.__setattr__(
        config,
        "speculative_config",
        _DSparkConfig(
            method="dspark",
            num_speculative_tokens=NUM_SPECULATIVE_TOKENS,
            speculator_config=SimpleNamespace(num_prefill_lookahead_tokens=1),
        ),
    )
    return config


def _dsv41_specs(config: VllmConfig):
    """{layer: spec} and {layer: backend}, in the model's registration order."""
    attn_cls = DeepseekV4FlashInferSM120Attention
    cache_config = config.cache_config
    assert not dsv41_attention._use_v41_mxfp8_kv_record()  # 584 B record off SM100
    specs, backends = {}, {}

    def add(name, backend, spec, is_draft=False):
        if isinstance(spec, AttentionSpec):
            spec = customize_attention_spec(backend, spec)
        if is_draft:
            spec = replace(spec, is_draft=True)
        specs[name], backends[name] = spec, backend

    for layer_id in range(HF.num_hidden_layers + HF.num_nextn_predict_layers):
        is_draft = layer_id >= HF.num_hidden_layers
        prefix = f"{'draft_model.' if is_draft else ''}model.layers.{layer_id}.attn"
        ratio = HF.compress_ratios[layer_id]
        is_kv_source = not is_draft and layer_id in HF.kv_source_layer_ids
        if is_kv_source:
            layer = SimpleNamespace(
                is_kv_source=True,
                kv_cache_dtype="fp8_ds_mla",
                kv_cache_torch_dtype=torch.uint8,
                head_dim=HF.head_dim,
                compress_ratio=ratio,
                kv_page_alignment=576,
                compressed_bytes_per_token=584,
            )
            add(prefix, attn_cls.backend_cls, attn_cls.get_kv_cache_spec(layer, config))
            index_cache = SimpleNamespace(
                cache_config=cache_config,
                head_dim=dsv41_attention._indexer_k_cache_head_dim(
                    HF.index_head_dim, False
                ),
                dtype=torch.uint8,
                compress_ratio=ratio,
                sparse_logits=False,
                rocm_mxfp4=False,
            )
            add(
                f"{prefix}.indexer.k_cache",
                DeepseekV41IndexerBackend,
                dsv41_attention.DeepseekV4IndexerCache.get_kv_cache_spec(
                    index_cache, config
                ),
            )
        swa_cache = SimpleNamespace(
            cache_config=cache_config,
            bounded_replay=True,
            block_size=attn_cls.swa_cache_block_size,
            head_dim=HF.head_dim,
            dtype=torch.uint8,
            window_size=HF.sliding_window,
            packed_bytes_per_token=584,
            packed_page_alignment=576,
            packed_block_stride_alignment=None,
        )
        add(
            f"{prefix}.swa_cache",
            DeepseekSparseSWAFlashInferBackend,
            DeepseekV4SWACache.get_kv_cache_spec(swa_cache, config),
            is_draft=is_draft,
        )
        if is_kv_source and ratio > 1:
            rows_per_step = NUM_SPECULATIVE_TOKENS + 2
            state_cache = SimpleNamespace(
                block_size=max(8, 1 << (rows_per_step - 1).bit_length()),
                state_dim=2 * HF.head_dim,
                dtype=torch.float32,
            )
            add(
                f"{prefix}.compressor.state_cache",
                CompressorBackend,
                CompressorStateCache.get_kv_cache_spec(state_cache, config),
            )
    return specs, backends


def _plan(config, specs, available_memory):
    groups = planning.get_global_kv_cache_groups(config, [specs])
    (kv_cache_config,) = planning.get_kv_cache_configs(
        config, [specs], [available_memory], groups
    )
    return kv_cache_config


@pytest.mark.parametrize(
    ("kv_cache_gib", "expected_tokens", "expected_concurrency"),
    [(8, 1_570_642, 1.50), (12, 2_355_983, 2.25), (20, 3_926_665, 3.74)],
)
def test_dsv41_sm12x_kv_pool_holds_the_qualified_token_count(
    sm12x, kv_cache_gib, expected_tokens, expected_concurrency
):
    config = _vllm_config()
    specs, _ = _dsv41_specs(config)
    assert len(specs) == 54  # 43 SWA + 4 latent + 4 index + 3 compressor rings

    # Raises "To serve at least one request ... 30.96 GiB KV cache is needed"
    # when the block stride is padded for 64-token kernel blocks.
    kv_cache_config = _plan(config, specs, kv_cache_gib * GiB)

    assert {t.block_stride for t in kv_cache_config.kv_cache_tensors} == {
        GROUP_PAGE_BYTES
    }
    assert kv_cache_config.num_blocks == kv_cache_gib * GiB // GROUP_PAGE_BYTES
    tokens, concurrency = get_kv_cache_capacity(config, kv_cache_config)
    assert tokens == expected_tokens
    assert round(concurrency, 2) == expected_concurrency


def test_dsv41_sm12x_packed_latent_blocks_bind_unsplit(sm12x):
    """The worker must be able to bind the planned pool: the packed 128-token
    latent blocks are addressed whole, never as 64-token kernel blocks."""
    config = _vllm_config()
    specs, backends = _dsv41_specs(config)
    groups = planning.get_global_kv_cache_groups(config, [specs])
    builder = planning.get_kv_cache_config_builder(config)
    kv_cache_config = builder.get_kv_cache_config_from_groups(config, groups, 16)
    attn_groups = [
        [
            AttentionGroup(backends[name], [name], specs[name], gid)
            for name in group.layer_names
        ]
        for gid, group in enumerate(kv_cache_config.kv_cache_groups)
    ]
    kernel_sizes = prepare_kernel_block_sizes(kv_cache_config, attn_groups)
    views = allocate_kv_cache(
        kv_cache_config, torch.device("cpu"), KVCacheLayout.BLHNC, kernel_sizes
    )

    split = {}
    for gid, layer_groups in enumerate(attn_groups):
        for group in layer_groups:
            (name,) = group.layer_names
            spec = group.kv_cache_spec
            if not isinstance(spec, AttentionSpec):
                continue
            # AttentionGroup.create_metadata_builders, minus the builders.
            kernel_block_size = select_common_block_size(
                kernel_sizes[gid], [group.backend], [spec]
            )
            if kernel_block_size != spec.block_size:
                group.kernel_block_size = kernel_block_size
                # Raises "... needs 64-token kernel blocks, which packed
                # 128-token KV cache blocks cannot be split into" when the
                # backend rejects the 128-token block.
                group.map_kv_cache_to_kernel_blocks(views[name])
                split[name] = kernel_block_size
    assert split == {}

    for layer_id in HF.kv_source_layer_ids:
        view = views[f"model.layers.{layer_id}.attn"]
        states = 128 // HF.compress_ratios[layer_id]
        assert view.shape == (16, 1, states, 584)
        assert view.stride(0) == GROUP_PAGE_BYTES
