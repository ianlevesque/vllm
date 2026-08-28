# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 implementation variant for ``FLASHINFER_MLA_SPARSE_SM120``."""

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
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer


_glm53_v2_warmup_active = False


def set_glm53_v2_warmup_active(active: bool) -> None:
    """Mark scheduler-realistic startup calls that need explicit GPU ordering."""
    global _glm53_v2_warmup_active
    _glm53_v2_warmup_active = active


def _kv_scale_format_for_model(model_type: str | None) -> str:
    if model_type is not None and model_type.startswith("glm"):
        return "arbitrary_fp32"
    return "pow2_fp32"


class FlashInferMLASparseSM120Impl(MLAAttentionImpl[FlashInferMLASparseMetadata]):
    """SM120 FlashInfer sparse-MLA implementation."""

    is_sparse = True

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

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        """Populate GLM's native-NoPE cache in the packed fp8_ds_mla layout."""
        if kv_cache.numel() == 0:
            return

        if not (
            kv_cache_dtype == "fp8_ds_mla"
            and self.kv_lora_rank == 512
            and self.qk_rope_head_dim == 0
        ):
            return super().do_kv_cache_update(
                kv_c_normed,
                k_pe,
                kv_cache,
                slot_mapping,
                kv_cache_dtype,
                k_scale,
            )

        if kv_cache.shape[-1] != 656:
            raise RuntimeError(
                "GLM native-NoPE SM121 path expected a 656-byte fp8_ds_mla "
                f"cache entry, got shape={tuple(kv_cache.shape)}"
            )
        if k_pe.shape[-1] != 0:
            raise RuntimeError(
                "GLM native-NoPE SM121 path requires an empty RoPE component; "
                f"got k_pe shape={tuple(k_pe.shape)}"
            )

        num_tokens = kv_c_normed.shape[0]
        if num_tokens == 0:
            return

        # Startup/profile dummy runs intentionally omit attention metadata and
        # fill slot_mapping with -1. The stock cache writer treats this as a
        # no-op, so avoid allocating the compatibility tail for that path.
        from vllm.forward_context import get_forward_context

        if get_forward_context().attn_metadata is None:
            return

        # concat_and_cache_mla requires a 64-element RoPE field for the packed
        # 656-byte layout. GLM is native NoPE, so materialize that field as
        # zeros; it contributes exactly zero to attention.
        zero_k_pe = torch.zeros(
            (num_tokens, 1, 64),
            dtype=kv_c_normed.dtype,
            device=kv_c_normed.device,
        )
        return super().do_kv_cache_update(
            kv_c_normed,
            zero_k_pe,
            kv_cache if kv_cache.dtype == torch.uint8 else kv_cache.view(torch.uint8),
            slot_mapping,
            kv_cache_dtype,
            k_scale,
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

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

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

        output = q.new_empty(
            (num_actual_toks, self.num_heads, self.kv_lora_rank),
            dtype=q.dtype,
        )

        if self.qk_rope_head_dim == 0 and self.kv_lora_rank == 512:
            if q.ndim != 3 or q.shape[-2:] != (4, 512):
                raise RuntimeError(
                    "GLM native-NoPE SM121 path expected four local TP16 query "
                    f"heads of width 512, got shape={tuple(q.shape)}"
                )
            if topk_indices_physical.ndim != 2:
                raise RuntimeError(
                    "GLM native-NoPE SM121 path expected a two-dimensional "
                    f"sparse index tensor, got shape={tuple(topk_indices_physical.shape)}"
                )
            if topk_indices_physical.shape[-1] < 2048:
                raise RuntimeError(
                    "GLM native-NoPE SM121 path expected the complete sparse row, "
                    f"got shape={tuple(topk_indices_physical.shape)}"
                )

            packed_cache = kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1)
            if (
                packed_cache.ndim != 4
                or packed_cache.shape[1] != 1
                or packed_cache.shape[-1] != 656
                or packed_cache.shape[2] <= 0
                or not packed_cache.is_contiguous()
            ):
                raise RuntimeError(
                    "GLM native-NoPE SM121 path expected contiguous HND "
                    "fp8_ds_mla cache pages, "
                    f"got shape={tuple(packed_cache.shape)}, "
                    f"contiguous={packed_cache.is_contiguous()}"
                )

            # The portable Triton kernel consumes global physical slot IDs and
            # flattens the cache itself. Preserve the complete sparse row and
            # 128-token physical page layout, and present native NoPE through
            # the packed format's all-zero 64-wide RoPE facade.
            from vllm.v1.attention.backends.mla.sm12x_sparse_mla_attn import (
                flash_mla_with_kvcache_triton,
            )

            padded_query = torch.nn.functional.pad(q.unsqueeze(1), (0, 64), value=0.0)
            if _glm53_v2_warmup_active:
                torch.cuda.synchronize(q.device)
            result, _lse = flash_mla_with_kvcache_triton(
                q=padded_query,
                k_cache=packed_cache,
                block_table=None,
                head_dim_v=512,
                softmax_scale=self.scale,
                causal=False,
                is_fp8_kvcache=True,
                indices=topk_indices_physical.unsqueeze(1),
                out=output.unsqueeze(1),
            )
            if _glm53_v2_warmup_active:
                torch.cuda.synchronize(q.device)
            return result.squeeze(1), None

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=None,
            max_seq_len=attn_metadata.topk_tokens,
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=attn_metadata.topk_tokens,
            kv_scale_format=self.kv_scale_format,
        )
        return out.squeeze(1), None
