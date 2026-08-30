# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 implementation variant for ``FLASHINFER_MLA_SPARSE_SM120``.

NoPE models (qk_rope_head_dim == 0, e.g. GLM-5.3-Flash) are supported by
zero-padding the rope section: the fp8_ds_mla cache layout is a fixed 656-byte
DeepSeek-shaped tile (512-dim fp8 latent + scales + 64-dim bf16 rope) and the
compiled ``concat_and_cache_mla`` kernel asserts pe_dim == 64. The padding is
applied symmetrically on both the KV-write and the query side; a zero rope
vector contributes exactly 0 to the q_pe . k_pe dot product, so attention
scores are bit-for-bit identical to true NoPE. This can be removed if the
kernels grow native pe_dim == 0 support.
"""

from typing import TYPE_CHECKING, cast

import torch

from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    MLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
    triton_filter_and_convert_dcp_index,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

_DS_ROPE_DIM = 64  # rope section width of the fixed 656-byte fp8_ds_mla tile
_SM120_MIN_KERNEL_HEADS = 8
_SM120_DECODE_MAX_TOKENS = 64
_SM120_DECODE_PAGE_SIZE = 64


def _pad_query_heads_for_sm120(q: torch.Tensor) -> torch.Tensor:
    """Pad small TP shards to FlashInfer's minimum instantiated head count."""
    num_heads = q.shape[1]
    if num_heads >= _SM120_MIN_KERNEL_HEADS:
        return q
    return torch.nn.functional.pad(
        q,
        (0, 0, 0, _SM120_MIN_KERNEL_HEADS - num_heads),
    )


def _reshape_kv_cache_for_sm120_decode(kv_cache: torch.Tensor) -> torch.Tensor:
    """Expose physical cache blocks as the decode kernel's 64-token pages."""
    packed = kv_cache.view(torch.uint8)
    if packed.ndim != 3 or packed.shape[-1] != 656:
        raise ValueError(
            "SM120 sparse MLA decode expects packed KV cache shape "
            f"[num_blocks, block_size, 656], got {tuple(packed.shape)}"
        )
    block_size = packed.shape[1]
    if block_size % _SM120_DECODE_PAGE_SIZE != 0:
        raise ValueError(
            "SM120 sparse MLA decode requires the KV-cache block size to be "
            f"divisible by {_SM120_DECODE_PAGE_SIZE}; got {block_size}"
        )
    return packed.reshape(-1, _SM120_DECODE_PAGE_SIZE, packed.shape[-1])


def _kv_scale_format_for_model(model_type: str | None) -> str:
    if model_type is not None and model_type.startswith("glm"):
        return "arbitrary_fp32"
    return "pow2_fp32"


class FlashInferMLASparseSM120Impl(MLAAttentionImpl[FlashInferMLASparseMetadata]):
    """SM120 FlashInfer sparse-MLA implementation."""

    is_sparse = True
    supports_dense_mha_prefill = False
    # Both decode return paths produce a base-2 LSE: the direct
    # sparse_mla_sm120_decode_dsv3_2 kernel scales scores by LOG2E and the
    # shared split-K merge writes log2f(sum) + max (FlashInfer
    # decode_dsv4_kernel.cuh), matching the trtllm-gen wrapper convention the
    # generic FlashInfer sparse impl declares.
    can_return_lse_for_decode: bool = True
    lse_base_on_e: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 does not support alibi_slopes / "
                "sliding_window / logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 only supports decoder self-attention"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        if self.kv_cache_dtype != "fp8_ds_mla":
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 requires the packed fp8_ds_mla "
                f"KV cache layout; got kv_cache_dtype={kv_cache_dtype!r}."
            )

        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim: int = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]
        # NoPE models (GLM-5.3): pad the rope section with zeros to fit the
        # DS-shaped tile. Exact: zero rope contributes nothing to scores.
        self._nope_pad = self.qk_rope_head_dim == 0
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        model_type = None
        if vllm_config.model_config is not None:
            model_type = getattr(
                vllm_config.model_config.hf_text_config, "model_type", None
            )
        self.kv_scale_format = _kv_scale_format_for_model(model_type)

        # Skip-topk layers are built with indexer=None and get the shared
        # buffer via mla_args instead (cf. FLASHMLA_SPARSE).
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer
            if indexer is not None
            else mla_args.get("topk_indices_buffer")
        )
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            raise RuntimeError(
                "FLASHINFER_MLA_SPARSE_SM120 requires FlashInfer's "
                "sparse MLA decode API."
            )
        assert self.topk_indices_buffer is not None

        self.supports_quant_query_input = False
        self._workspace_buffer: torch.Tensor | None = None
        self._decode_lse_buffer: torch.Tensor | None = None

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        if self._nope_pad and k_pe.size(-1) == 0:
            k_pe = k_pe.new_zeros((*k_pe.shape[:-1], _DS_ROPE_DIM))
        from vllm import _custom_ops as ops

        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]

        rope_dim = self.qk_rope_head_dim
        if self._nope_pad:
            q = torch.nn.functional.pad(q, (0, _DS_ROPE_DIM))
            rope_dim = _DS_ROPE_DIM

        # FlashInfer's SM120 DSv3.2/GLM decode and prefill dispatch tables are
        # instantiated for 8 local query heads and above. GLM-5.3-Flash has 64
        # global heads, so TP16 produces four local heads and otherwise falls
        # through to the prefill orchestrator even for decode-shaped batches.
        # Sparse MLA is independent per query head: zero-pad the query tile to
        # the minimum supported width and discard the added outputs below.
        # Under DCP the incoming query is already head-gathered across the DCP
        # group (dcp_world_size * local heads) and the shared reducer
        # reduce-scatters that same head set back, so the returned width must
        # be the incoming width, not self.num_heads.
        in_num_heads = q.shape[1]
        q = _pad_query_heads_for_sm120(q)
        kernel_num_heads = q.shape[1]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        empty_rows: torch.Tensor | None = None
        if self.dcp_world_size > 1:
            # Each rank holds only its interleaved KV shard: drop indices owned
            # by other ranks, compact this rank's slots to a contiguous prefix
            # and get the per-token valid counts for the kernel.
            topk_indices_physical, seq_lens = triton_filter_and_convert_dcp_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=(attn_metadata.cp_kv_cache_interleave_size),
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )
            empty_rows = seq_lens == 0
        else:
            topk_indices_physical = cast(
                torch.Tensor,
                triton_convert_req_index_to_global_index(
                    attn_metadata.req_id_per_token[:num_actual_toks],
                    attn_metadata.block_table,
                    topk_indices,
                    BLOCK_SIZE=attn_metadata.block_size,
                    NUM_TOPK_TOKENS=topk_indices.shape[1],
                ),
            )
            seq_lens = None

        output = q.new_empty(
            (num_actual_toks, kernel_num_heads, self.kv_lora_rank),
            dtype=q.dtype,
        )

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        # The indexer's kpool can widen the index buffer past the configured
        # index_topk (always-selected tail slots, padded up by the buffer
        # allocation). The kernel is generic over the width, but its shape
        # check requires sparse_mla_top_k == the indices' actual width — pass
        # the buffer width, not attn_metadata.topk_tokens.
        eff_topk = topk_indices_physical.shape[-1]

        # FlashInfer's public sparse-MLA wrapper dispatches decode only for a
        # fixed shape table and otherwise falls through to the prefill
        # orchestrator. The latter deliberately rejects num_tokens <= 64,
        # turning an unsupported/misread predicate into an opaque C++ error.
        # Call the same DSv3.2/GLM decode kernel directly for decode-shaped
        # batches. This also makes the kernel's 64-token virtual KV pages
        # explicit instead of relying on the wrapper to infer them from the
        # cache view. MTP warmup reaches this path with exactly 64 tokens.
        if num_actual_toks <= _SM120_DECODE_MAX_TOKENS:
            if kernel_num_heads != _SM120_MIN_KERNEL_HEADS or eff_topk != 2048:
                raise ValueError(
                    "SM120 GLM sparse MLA decode requires 8 kernel query heads "
                    f"and effective topk 2048; got heads={kernel_num_heads}, "
                    f"topk={eff_topk}"
                )

            from flashinfer.mla._core import _sparse_mla_decode_workspace
            from flashinfer.mla._sparse_mla_sm120 import (
                _MODEL_TYPE_GLM_NSA,
                sparse_mla_sm120_decode_dsv3_2,
            )

            mid_out, mid_lse = _sparse_mla_decode_workspace(
                self._workspace_buffer,
                num_tokens=num_actual_toks,
                num_heads=kernel_num_heads,
                d_v=self.kv_lora_rank,
                topk=eff_topk,
                extra_topk=0,
            )
            if mid_out is None or mid_lse is None:
                raise RuntimeError(
                    "FlashInfer workspace is too small for SM120 sparse MLA decode"
                )
            if (
                self._decode_lse_buffer is None
                or self._decode_lse_buffer.shape[0] < num_actual_toks
                or self._decode_lse_buffer.shape[1] < kernel_num_heads
            ):
                self._decode_lse_buffer = torch.empty(
                    (_SM120_DECODE_MAX_TOKENS, kernel_num_heads),
                    dtype=torch.float32,
                    device=q.device,
                )

            out_lse = self._decode_lse_buffer[:num_actual_toks, :kernel_num_heads]
            out = sparse_mla_sm120_decode_dsv3_2(
                q=q,
                kv_cache=_reshape_kv_cache_for_sm120_decode(kv_c_and_k_pe_cache),
                indices=topk_indices_physical,
                mid_out=mid_out,
                mid_lse=mid_lse,
                output=output,
                out_lse=out_lse,
                sm_scale=self.scale,
                model_type=_MODEL_TYPE_GLM_NSA,
                # Bypass autotuning and let the compiled kernel use its
                # occupancy-aware heuristic. The fleet disables FI autotune.
                chunks_per_block=-1,
            )
            out = out[:, :in_num_heads].contiguous()
            if not self.need_to_return_lse_for_decode:
                return out, None
            # The kernel fills out_lse (base-2) as a side effect; hand it to
            # the DCP reducer. Rows whose top-k slots all live on other ranks
            # write -1e30 split LSEs but the split merge accumulates their
            # scratch unguarded, so mask them out explicitly.
            lse = out_lse[:, :in_num_heads]
            if empty_rows is not None:
                out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
                lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))
            return out, lse

        kernel_out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=rope_dim,
            block_tables=topk_indices_physical.unsqueeze(1),
            # Compacted per-token valid counts under DCP (the SM120 route maps
            # seq_lens to the kernel's per-token topk_length); None keeps the
            # uniform-top-k behaviour otherwise.
            seq_lens=seq_lens,
            max_seq_len=eff_topk,
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=eff_topk,
            return_lse=self.need_to_return_lse_for_decode,
            kv_scale_format=self.kv_scale_format,
        )
        if self.need_to_return_lse_for_decode:
            assert isinstance(kernel_out, tuple)
            out, lse = kernel_out
        else:
            assert isinstance(kernel_out, torch.Tensor)
            out = kernel_out
            lse = None

        out = out.squeeze(1)[:, :in_num_heads].contiguous()
        if lse is None:
            return out, None
        lse = self._normalize_lse(lse, num_actual_toks, kernel_num_heads)
        lse = lse[:, :in_num_heads]
        if empty_rows is not None:
            out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
            lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))
        return out, lse

    @staticmethod
    def _normalize_lse(
        lse: torch.Tensor,
        num_tokens: int,
        num_heads: int,
    ) -> torch.Tensor:
        # FlashInfer returns the decode LSE either as 2D (num_tokens, num_heads)
        # or 3D ((num_tokens, num_heads, 1) / (num_tokens, 1, num_heads)).
        # Collapse all of these to the (num_tokens, num_heads) the shared DCP
        # reducer expects.
        if lse.dim() == 3:
            if lse.shape[-1] == 1:
                lse = lse.squeeze(-1)
            elif lse.shape[1] == 1:
                lse = lse.squeeze(1)
            elif lse.shape[0] * lse.shape[1] == num_tokens:
                lse = lse.reshape(num_tokens, lse.shape[-1])
        if lse.shape != (num_tokens, num_heads):
            raise RuntimeError(
                "Unexpected FlashInfer SM120 sparse MLA LSE shape: "
                f"{tuple(lse.shape)}, expected ({num_tokens}, {num_heads})."
            )
        return lse
