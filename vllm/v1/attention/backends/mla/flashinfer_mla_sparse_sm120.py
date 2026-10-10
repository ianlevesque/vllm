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
    triton_filter_and_convert_dcp_index,
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
    supports_dcp: bool = True
    can_return_lse_for_decode: bool = True
    # FlashInfer 0.7.0.post1's SM120 split-K merge returns log2 LSE.
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
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        index_group = self.index_group
        if isinstance(index_group, HiSparseMLAIndexGroup):
            assert self.dcp_world_size == 1
            num_decode_tokens = attn_metadata.num_decode_tokens
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
                        index_group.physical_kv_cache(self.index_group_index),
                        topk_indices_physical,
                    )[0]
                )
            if num_decode_tokens < num_actual_toks:
                cache = index_group.cache(self.index_group_index)
                if num_decode_tokens == 0 and cache.all_context_pages_resident:
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
                    prefill_cache = index_group.physical_kv_cache(
                        self.index_group_index
                    )
                else:
                    prefill_cache, block_table, req_ids = (
                        index_group.stage_prefill_rows(
                            self.index_group_index,
                            kv_c_and_k_pe_cache,
                            attn_metadata,
                        )
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
                    )[0]
                )
            output = torch.cat(outputs) if len(outputs) > 1 else outputs[0]
            return output, None

        kv_rows, block_stride_rows = flat_kv_row_view(
            kv_c_and_k_pe_cache, attn_metadata.block_size
        )
        if self.dcp_world_size > 1:
            topk_indices_physical, seq_lens = triton_filter_and_convert_dcp_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=attn_metadata.cp_kv_cache_interleave_size,
                BLOCK_SIZE=attn_metadata.block_size,
                BLOCK_STRIDE_ROWS=block_stride_rows,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                return_valid_counts=True,
            )
        else:
            topk_indices_physical = cast(
                torch.Tensor,
                triton_convert_req_index_to_global_index(
                    attn_metadata.req_id_per_token[:num_actual_toks],
                    attn_metadata.block_table,
                    topk_indices,
                    BLOCK_SIZE=attn_metadata.block_size,
                    BLOCK_STRIDE_ROWS=block_stride_rows,
                    NUM_TOPK_TOKENS=topk_indices.shape[1],
                ),
            )
            seq_lens = None
        return self._run_mqa_kernel(
            q,
            # The SM120 kernels are built for 64-row pages only. Keep the
            # upstream flat row view: physical indices include padded strides.
            kv_rows.view(-1, 64, kv_rows.shape[-1]),
            topk_indices_physical,
            seq_lens,
        )

    def _run_mqa_kernel(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_indices_physical: torch.Tensor,
        seq_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_actual_toks, input_heads = q.shape[:2]
        # The shared FlashInfer version accepts arbitrary decode head counts,
        # but its prefill path starts at eight. TP16 GLM has four local heads;
        # DCP queries already contain the gathered heads, which must survive
        # until the common DCP reducer scatters them back to their owners.
        kernel_heads = max(8, input_heads)
        if kernel_heads != input_heads:
            q = torch.nn.functional.pad(q, (0, 0, 0, kernel_heads - input_heads))

        output = q.new_empty(
            (num_actual_toks, kernel_heads, self.kv_lora_rank),
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
            seq_lens=seq_lens,
            max_seq_len=sparse_capacity,
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=sparse_capacity,
            kv_scale_format=self.kv_scale_format,
            return_lse=self.need_to_return_lse_for_decode,
        )
        if self.need_to_return_lse_for_decode:
            out, lse = out
            if lse.ndim == 3 and lse.shape[1] == 1:
                lse = lse.squeeze(1)
            elif lse.ndim == 3 and lse.shape[-1] == 1:
                lse = lse.squeeze(-1)
            if lse.shape != (num_actual_toks, kernel_heads):
                raise RuntimeError(
                    f"Unexpected SM120 sparse MLA LSE shape: {lse.shape}"
                )
            lse = lse[:, :input_heads].contiguous()
        else:
            lse = None
        out = out.squeeze(1)[:, :input_heads].contiguous()
        if seq_lens is not None:
            # A DCP shard may own no selected tokens. Its kernel scratch is
            # not a valid partial softmax; use the reducer's neutral element.
            empty_rows = seq_lens == 0
            out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
            if lse is not None:
                lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))
        return out, lse
