# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 DSA indexer cache helpers (fleet port of PR #65685).

The upstream packed-cache classes (KimiK3IndexerCache/Backend and
packed_dsa_block_size) target a newer-main KV-cache API that this
lineage does not have; the fleet port uses the stock
DeepseekV32IndexerCache with its standard MLAAttentionSpec instead.
Only the fp4-selection and entry-size helpers are needed here.
"""

from vllm.config import VllmConfig
from vllm.model_executor.layers.sparse_attn_indexer import MXFP4_BLOCK_SIZE
from vllm.v1.attention.backends.mla.indexer import dsa_indexer_uses_fp4


def kimi_k3_indexer_uses_fp4(vllm_config: VllmConfig) -> bool:
    """FP8 (132 B/token) indexer keys by default (``indexer_kv_dtype='auto'``);
    MXFP4 (68 B/token) only when requested with ``--attention-config
    '{"indexer_kv_dtype": "mxfp4"}'``. MXFP4 needs SM10x (checked by
    ``dsa_indexer_uses_fp4``), the fused key insert (layernorm k-norm, no indexer
    RoPE) and no PCP; anything else raises instead of writing another format."""
    if not dsa_indexer_uses_fp4(vllm_config):
        return False
    text_config = getattr(vllm_config.model_config, "hf_text_config", None)
    dsa_config = getattr(text_config, "dsa_config", None)
    if dsa_config is not None and not (
        getattr(dsa_config, "index_k_norm_type", "rmsnorm") == "layernorm"
        and getattr(dsa_config, "index_rope_dim", 0) == 0
    ):
        raise ValueError(
            "indexer_kv_dtype='mxfp4' needs the fused Kimi K3 DSA key insert "
            "(index_k_norm_type='layernorm', index_rope_dim=0)."
        )
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if getattr(parallel_config, "prefill_context_parallel_size", 1) > 1:
        raise ValueError(
            "indexer_kv_dtype='mxfp4' is not supported with prefill context "
            "parallelism for Kimi K3 DSA."
        )
    return True


def indexer_cache_entry_bytes(head_dim: int, use_fp4_cache: bool) -> int:
    """Indexer key cache bytes per token: FP8 values + one 4 B scale, or MXFP4
    values (two per byte) + one UE8M0 byte per 32 values."""
    if use_fp4_cache:
        return head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
    return head_dim + 4
