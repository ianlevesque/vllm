# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 implementation variant for ``FLASHINFER_MLA_SPARSE_SM120``."""

from typing import TYPE_CHECKING, cast

import torch

from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
)
from vllm.v1.attention.backend import AttentionLayer, AttentionType
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    triton_convert_req_index_to_global_index,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer


def _kv_scale_format_for_model(model_type: str | None) -> str:
    if model_type is not None and model_type.startswith("glm"):
        return "arbitrary_fp32"
    return "pow2_fp32"


class FlashInferMLASparseSM120Impl(SparseMLACommonImpl[FlashInferMLASparseMetadata]):
    """SM120 FlashInfer sparse-MLA implementation."""

    is_sparse = True
    supports_dense_mha_prefill = False
    # The trtllm-gen return path produces a base-2 LSE, matching the
    # generic FlashInfer sparse impl's declaration. DCP support rides
    # the common sparse path (index filtering + LSE combine); adapted
    # from upstream PR vllm-project/vllm#47779 via the fleet's
    # unified/2026-08-dcp carry for GLM TP16+DCP2.
    can_return_lse_for_decode: bool = True
    supports_dcp: bool = True
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

        if kv_cache_dtype != "fp8_ds_mla":
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 requires the packed fp8_ds_mla "
                f"KV cache layout; got kv_cache_dtype={kv_cache_dtype!r}."
            )

        topk_indices_buffer = mla_args.pop("topk_indices_buffer", None)
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        model_type = None
        if vllm_config.model_config is not None:
            model_type = getattr(
                vllm_config.model_config.hf_text_config, "model_type", None
            )
        self.kv_scale_format = _kv_scale_format_for_model(model_type)

        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            raise RuntimeError(
                "FLASHINFER_MLA_SPARSE_SM120 requires FlashInfer's "
                "sparse MLA decode API."
            )
        assert self.topk_indices_buffer is not None

        self.supports_quant_query_input = False
        self._workspace_buffer: torch.Tensor | None = None

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not isinstance(self.index_group, HiSparseMLAIndexGroup):
            # The common sparse path handles DCP top-k filtering and
            # valid-count computation, then calls _forward_mqa_kernel
            # below. HiSparse keeps its dedicated path here (HiSparse
            # does not support DCP).
            return super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        index_group = self.index_group
        if isinstance(index_group, HiSparseMLAIndexGroup):
            num_decode_tokens = attn_metadata.num_decode_tokens
            physical_kv_cache = index_group.physical_kv_cache(self.index_group_index)
            if (
                num_decode_tokens == num_actual_toks
                or index_group.cache(self.index_group_index).all_context_pages_resident
            ):
                topk_indices_physical = cast(
                    torch.Tensor,
                    index_group.convert_logical_to_physical_topk(
                        self.index_group_index,
                        topk_indices,
                        attn_metadata,
                        block_stride_rows=None,
                        return_valid_counts=False,
                    ),
                )
                return (
                    self._run_mqa_kernel(q, physical_kv_cache, topk_indices_physical),
                    None,
                )

            outputs = []
            if num_decode_tokens:
                topk_indices_physical = cast(
                    torch.Tensor,
                    index_group.convert_logical_to_physical_topk(
                        self.index_group_index,
                        topk_indices[:num_decode_tokens],
                        attn_metadata,
                        block_stride_rows=None,
                        return_valid_counts=False,
                    ),
                )
                outputs.append(
                    self._run_mqa_kernel(
                        q[:num_decode_tokens],
                        physical_kv_cache,
                        topk_indices_physical,
                    )
                )
            prefill_cache, block_table, req_ids = index_group.stage_prefill_rows(
                self.index_group_index,
                kv_c_and_k_pe_cache,
                attn_metadata,
            )
            topk_indices_physical = cast(
                torch.Tensor,
                triton_convert_req_index_to_global_index(
                    req_ids,
                    block_table,
                    topk_indices[num_decode_tokens:],
                    BLOCK_SIZE=attn_metadata.block_size,
                    NUM_TOPK_TOKENS=topk_indices.shape[1],
                ),
            )
            outputs.append(
                self._run_mqa_kernel(
                    q[num_decode_tokens:],
                    prefill_cache,
                    topk_indices_physical,
                )
            )
            output = torch.cat(outputs) if len(outputs) > 1 else outputs[0]
            return output, None

    def _run_mqa_kernel(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_indices_physical: torch.Tensor,
    ) -> torch.Tensor:
        num_actual_toks = q.shape[0]

        output = q.new_empty(
            (num_actual_toks, self.num_heads, self.kv_lora_rank),
            dtype=q.dtype,
        )

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        sparse_capacity = topk_indices_physical.shape[1]
        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=None,
            max_seq_len=sparse_capacity,
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=sparse_capacity,
            kv_scale_format=self.kv_scale_format,
        )
        return out.squeeze(1)

    def _forward_mqa_kernel(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        valid_counts: torch.Tensor,
        *,
        layer: AttentionLayer,
        block_size: int,
        is_decode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        num_actual_toks = q.shape[0]
        # Under DCP the incoming query is already head-gathered across
        # the DCP group (dcp_world_size * local heads) and the shared
        # reducer reduce-scatters that same head set back, so the
        # returned width must be the incoming width, not
        # self.num_heads.
        in_num_heads = q.shape[1]

        kv_rows, _ = flat_kv_row_view(kv_cache, block_size)
        # The SM120 kernels are built for 64-row pages only.
        kv_cache = kv_rows.view(-1, 64, kv_rows.shape[-1])

        output = q.new_empty(
            (num_actual_toks, in_num_heads, self.kv_lora_rank),
            dtype=q.dtype,
        )
        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        sparse_capacity = topk_indices.shape[1]
        # Under DCP each rank holds only its interleaved KV shard; the
        # common path already compacted this rank's top-k slots and
        # counted the valid prefix per token. The SM120 route maps
        # seq_lens to the kernel's per-token topk_length. Keep the
        # uniform-top-k behaviour (seq_lens=None) otherwise.
        seq_lens = valid_counts if self.dcp_world_size > 1 else None
        kernel_out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=topk_indices.unsqueeze(1),
            seq_lens=seq_lens,
            max_seq_len=sparse_capacity,
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=sparse_capacity,
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
        out = out.squeeze(1)
        if lse is None:
            return out, None
        lse = self._normalize_lse(lse, num_actual_toks, in_num_heads)
        # Rows whose top-k slots all live on other DCP ranks must not
        # contribute to the cross-rank merge.
        empty_rows = valid_counts == 0
        out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
        lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))
        return out, lse

    @staticmethod
    def _normalize_lse(
        lse: torch.Tensor,
        num_tokens: int,
        num_heads: int,
    ) -> torch.Tensor:
        # FlashInfer returns the decode LSE either as 2D (num_tokens,
        # num_heads) or 3D ((num_tokens, num_heads, 1) /
        # (num_tokens, 1, num_heads)). Collapse all of these to the
        # (num_tokens, num_heads) the shared DCP reducer expects.
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
