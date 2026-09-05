# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.utils.import_utils import has_deep_gemm
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.compressor_utils import (
    CompressedSlotMappingKernel,
)
from vllm.v1.attention.backends.mla.indexer import (
    BuildPrefillChunkMetadataKernel,
    DeepseekV32IndexerMetadataBuilder,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    ConvertReqIndexToGlobalIndexKernel,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.worker.block_table import get_block_table_width


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_deep_gemm(),
    reason="requires CUDA and DeepGEMM metadata kernels",
)
@pytest.mark.parametrize("capture_query_lens", [[1, 1, 1, 1], [4]])
def test_compressed_flattened_decode_graph_reads_current_lengths(
    monkeypatch, capture_query_lens
):
    """One-token capture must observe later speculative lengths and padding.

    A different context buffer at replay disagrees with fresh DeepGEMM schedule
    metadata and can deadlock at a KV split boundary. Capture only a tensor copy
    so the regression fails safely instead of launching that inconsistent pair.
    """
    from vllm.v1.attention.backends.mla import indexer

    monkeypatch.setattr(indexer, "_use_flattening", lambda _: True)
    monkeypatch.setattr(indexer, "_supports_varlen_paged_mqa_logits", lambda: False)
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=2048),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64, max_num_seqs=16),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        num_speculative_tokens=3,
        speculative_config=SimpleNamespace(
            num_speculative_tokens=3, enable_adaptive_verification=False
        ),
        attention_config=SimpleNamespace(resolve_indexer_kv_dtype=lambda _: "fp8"),
    )
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )
    builder = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["dummy"],
        vllm_config=config,
        device=torch.device("cuda"),
        block_table_width=8,
    )
    builder.set_kernel_block_size(256)

    def build(lengths, queries):
        common = create_common_attn_metadata(
            BatchSpec(lengths, queries), block_size=256, device=torch.device("cuda")
        )
        common.block_table_tensor = torch.nn.functional.pad(
            common.block_table_tensor, (0, 8 - common.block_table_tensor.shape[1])
        )
        return builder.build(0, common).decode

    with set_current_vllm_config(VllmConfig()):
        captured = build(capture_query_lens, capture_query_lens)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            observed = captured.seq_lens.clone()
        for lengths, queries, expected in [
            ([516, 0, 0, 0], [4, 0, 0, 0], [128, 128, 128, 129]),
            ([1024, 512, 0, 0], [3, 1, 0, 0], [255, 255, 256, 128]),
            ([520, 0, 0, 0], [1, 1, 1, 1], [130, 0, 0, 0]),
        ]:
            current = build(lengths, queries)
            graph.replay()
            torch.accelerator.synchronize()
            assert current.seq_lens.flatten().tolist() == expected
            assert observed.flatten().tolist() == expected
            assert current.seq_lens.data_ptr() == captured.seq_lens.data_ptr()


def test_indexer_warmup_normalizes_zero_compress_ratios():
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(compress_ratios=[0, 0, 4, 128, 0], index_kpool=32)
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
    )

    keys = BuildPrefillChunkMetadataKernel().get_warmup_keys(config)

    assert {key.compress_ratio for key in keys} == {1, 4, 32, 128}
    assert {(key.query_slice_start, key.query_slice_stop) for key in keys} == {
        (query_slice_start, query_slice_stop)
        for query_slice_start in (1, 2, 16)
        for query_slice_stop in (1, 2, 16)
    }


def test_compressed_slot_mapping_warmup_includes_index_kpool():
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=256),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(index_kpool=32)),
    )

    keys = CompressedSlotMappingKernel().get_warmup_keys(config)
    assert {(key.compress_ratio, key.block_size) for key in keys} == {(32, 2)}


def test_index_conversion_warmup_uses_physical_block_stride():
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64),
        model_config=SimpleNamespace(
            max_model_len=1024,
            hf_config=SimpleNamespace(index_topk=2048),
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
    )

    keys = ConvertReqIndexToGlobalIndexKernel().get_warmup_keys(
        config,
        block_stride_rows=4096,
    )
    assert {key.block_stride_rows for key in keys} == {4096}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_indexer_builder_deepseek_v4_compressed_slot_mapping_uses_num_states():
    """Regression test: DeepseekV4 compression path must compute slot_mapping from
    compressed positions, not reuse the uncompressed common metadata mapping.
    """
    device = torch.device("cuda")

    # num_states = block_size // tokens_per_state = 256 // 4 = 64
    kv_cache_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )
    vllm_config = create_vllm_config(max_model_len=1024)
    max_num_blocks = kv_cache_spec.max_num_blocks_per_req(vllm_config, 1024)
    block_table_width = get_block_table_width(max_num_blocks, kv_cache_spec.block_size)
    builder = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=device,
        block_table_width=block_table_width,
    )

    # Construct a single request where:
    # - num_computed = 240 (=> compressed_pos_start = 60)
    # - query_len = 40 (=> num_groups = 10)
    # => compressed positions are 60..69 which cross the storage block boundary at 64.
    query_start_loc = torch.tensor([0, 40], dtype=torch.int32, device=device)
    query_start_loc_cpu = query_start_loc.cpu()
    seq_lens = torch.tensor([280], dtype=torch.int32, device=device)  # 240 + 40

    # Two blocks: compressed positions 0..63 map to block 5, 64..127 map to block 7.
    block_table_tensor = torch.tensor([[5, 7]], dtype=torch.int32, device=device)

    # Dummy uncompressed slot mapping (length == uncompressed num_actual_tokens).
    slot_mapping = torch.full((40,), -123, dtype=torch.int64, device=device)

    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens.cpu(),
        num_reqs=1,
        num_actual_tokens=40,
        max_query_len=40,
        max_seq_len=280,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
    )

    md = builder.build(common_prefix_len=0, common_attn_metadata=common)

    # The compressed slot_mapping retains the original uncompressed size (40).
    # Only every compress_ratio-th position gets a valid slot; the rest are -1.
    assert md.slot_mapping.numel() == 40
    valid_slots = md.slot_mapping[md.slot_mapping >= 0]
    assert valid_slots.numel() == 10  # 40 tokens / compress_ratio 4

    storage_bs = kv_cache_spec.num_states  # 64
    # Compressed positions 60..63 land in block 5, positions 64..69 in block 7.
    expected = torch.tensor(
        [
            5 * storage_bs + 60,
            5 * storage_bs + 61,
            5 * storage_bs + 62,
            5 * storage_bs + 63,
        ]
        + [
            7 * storage_bs + 0,
            7 * storage_bs + 1,
            7 * storage_bs + 2,
            7 * storage_bs + 3,
            7 * storage_bs + 4,
            7 * storage_bs + 5,
        ],
        dtype=torch.int64,
        device=device,
    )
    torch.testing.assert_close(valid_slots, expected)
