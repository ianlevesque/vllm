# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native b12x dense MLA decode backend for Kimi K3 on SM120/SM121."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch

from vllm.compilation.b12x_capture import is_b12x_compile_only_warmup
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import (
    dcp_a2a_lse_reduce,
    dcp_b12x_all_gather_heads,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

_K3_ABSORBED_HEAD_DIM = 576
_K3_KV_LORA_RANK = 512
_K3_QK_NOPE_HEAD_DIM = 128
_K3_QK_ROPE_HEAD_DIM = 64
_K3_QK_HEAD_DIM = 192
_K3_V_HEAD_DIM = 128
_MAX_B12X_QUERY_ROWS = 1024
_MAX_B12X_CACHE_TOKENS = 1_048_576
_B12X_QUERY_HEAD_TILE = 8
_MAX_I32 = torch.iinfo(torch.int32).max


def _load_dense_mla() -> Any:
    from b12x.attention import dense_mla

    return dense_mla


def _page_table_width(max_cache_tokens: int, page_size: int) -> int:
    width = (max_cache_tokens + page_size - 1) // page_size
    # vLLM pads block tables to a 128-token boundary for page sizes <= 128.
    if page_size <= 128:
        alignment = 128 // page_size
        width = ((width + alignment - 1) // alignment) * alignment
    return width


def _max_dcp_local_cache_tokens(
    vllm_config: VllmConfig, *, dcp_size: int | None = None
) -> int:
    """Return the largest token shard held by one decode-context rank."""
    parallel_config = vllm_config.parallel_config
    dcp_size = int(
        parallel_config.decode_context_parallel_size if dcp_size is None else dcp_size
    )
    interleave = int(parallel_config.cp_kv_cache_interleave_size)
    max_model_len = int(vllm_config.model_config.max_model_len)
    partitions = dcp_size * interleave
    return ((max_model_len + partitions - 1) // partitions) * interleave


def _planned_kv_dtype(vllm_config: VllmConfig) -> torch.dtype:
    cache_dtype = vllm_config.cache_config.cache_dtype
    if cache_dtype == "auto":
        return vllm_config.model_config.dtype
    if cache_dtype == "bfloat16":
        return torch.bfloat16
    if cache_dtype in ("fp8", "fp8_e4m3"):
        fp8_dtype = current_platform.fp8_dtype()
        if fp8_dtype != torch.float8_e4m3fn:
            raise ValueError(
                "B12X_MLA requires native E4M3 FP8 KV storage; "
                f"this platform selected {fp8_dtype}."
            )
        return fp8_dtype
    raise ValueError(
        f"B12X_MLA supports only BF16 or E4M3 KV cache storage, got {cache_dtype!r}."
    )


def _kernel_query_heads(local_heads: int, dcp_size: int) -> int:
    """Return the head count presented to the tiled dense-MLA kernel.

    B12X computes each query head independently but launches them in
    tiles of eight.  K3 has 96 heads, so TP16 without DCP produces six local
    heads.  Padding that DCP1 query to eight is mathematically inert and lets
    the native kernel cover this otherwise valid tensor-parallel layout.

    DCP reductions depend on the gathered head layout, so keep their existing
    exact-multiple contract rather than introducing synthetic collective
    entries.
    """
    if local_heads <= 0:
        raise ValueError(f"B12X_MLA requires positive local heads, got {local_heads}.")
    if dcp_size <= 0:
        raise ValueError(f"B12X_MLA requires positive DCP size, got {dcp_size}.")
    effective_heads = local_heads * dcp_size
    if dcp_size > 1:
        if effective_heads % _B12X_QUERY_HEAD_TILE:
            raise ValueError(
                "B12X_MLA requires a multiple of 8 query heads after DCP "
                f"gather, got local={local_heads}, DCP={dcp_size}, "
                f"effective={effective_heads}."
            )
        return effective_heads
    return (
        (effective_heads + _B12X_QUERY_HEAD_TILE - 1)
        // _B12X_QUERY_HEAD_TILE
        * _B12X_QUERY_HEAD_TILE
    )


def _active_dense_mla_splits(plan: Any, max_seq_len: int | None) -> int:
    """Return the useful prefix of a capture-static dense-MLA split plan.

    The plan's split boundaries and scratch layout remain fixed at their
    maximum-context values.  Short decode steps only omit tail splits whose
    first 64-token chunk is beyond every live sequence, so this does not alter
    the reduction order of any contributing partial.
    """
    num_splits = int(getattr(plan, "num_splits", 1))
    chunks_per_split = int(getattr(plan, "chunks_per_split", 1))
    if num_splits <= 0 or chunks_per_split <= 0:
        raise ValueError(
            "B12X_MLA received an invalid dense MLA split plan: "
            f"num_splits={num_splits}, chunks_per_split={chunks_per_split}."
        )
    if max_seq_len is None:
        return num_splits
    valid_chunks = max(1, (max(0, int(max_seq_len)) + 63) // 64)
    return min(
        num_splits,
        (valid_chunks + chunks_per_split - 1) // chunks_per_split,
    )


def _dcp_local_seq_lens_from_global(
    output: torch.Tensor,
    remainder_scratch: torch.Tensor,
    global_seq_lens: torch.Tensor,
    *,
    dcp_size: int,
    dcp_rank: int,
    interleave: int,
) -> None:
    """Convert global lengths to this rank's round-robin DCP lengths.

    This is the allocation-free tensor equivalent of
    ``prepare_dcp_local_seq_lens``.  DSpark target verification flattens one
    causal query block into several independent decode rows, so every row
    needs the local length corresponding to its own global token position.
    """
    if output.shape != global_seq_lens.shape or output.shape != remainder_scratch.shape:
        raise ValueError(
            "B12X_MLA DCP sequence-length buffers must have identical shapes, "
            f"got output={output.shape}, scratch={remainder_scratch.shape}, "
            f"global={global_seq_lens.shape}."
        )
    if output.dtype != torch.int32 or remainder_scratch.dtype != torch.int32:
        raise TypeError("B12X_MLA DCP sequence-length buffers must use int32.")
    if dcp_size <= 0 or not 0 <= dcp_rank < dcp_size or interleave <= 0:
        raise ValueError(
            "Invalid B12X_MLA DCP layout: "
            f"size={dcp_size}, rank={dcp_rank}, interleave={interleave}."
        )
    if dcp_size == 1:
        output.copy_(global_seq_lens)
        return

    virtual_block = dcp_size * interleave
    torch.div(global_seq_lens, virtual_block, rounding_mode="floor", out=output)
    output.mul_(interleave)
    torch.remainder(global_seq_lens, virtual_block, out=remainder_scratch)
    remainder_scratch.sub_(dcp_rank * interleave)
    remainder_scratch.clamp_(min=0, max=interleave)
    output.add_(remainder_scratch)


def _create_dense_mla_plan(
    vllm_config: VllmConfig,
    device: torch.device,
    *,
    page_size: int,
    num_q_heads: int,
    max_total_q: int | None = None,
    dcp_size: int | None = None,
) -> Any:
    dense_mla = _load_dense_mla()
    max_total_q = int(
        max_total_q
        if max_total_q is not None
        else vllm_config.scheduler_config.max_num_seqs
    )
    max_cache_tokens = _max_dcp_local_cache_tokens(vllm_config, dcp_size=dcp_size)
    if max_total_q > _MAX_B12X_QUERY_ROWS:
        raise ValueError(
            "B12X_MLA supports at most "
            f"{_MAX_B12X_QUERY_ROWS} simultaneous decode rows, got {max_total_q}."
        )
    if max_cache_tokens > _MAX_B12X_CACHE_TOKENS:
        raise ValueError(
            "B12X_MLA supports at most "
            f"{_MAX_B12X_CACHE_TOKENS} cache tokens, got {max_cache_tokens}."
        )

    caps = dense_mla.Caps(
        device=device,
        mode="decode",
        dtype=torch.bfloat16,
        kv_dtype=_planned_kv_dtype(vllm_config),
        num_q_heads=num_q_heads,
        page_size=page_size,
        max_total_q=max_total_q,
        max_batch=max_total_q,
        max_cache_tokens=max_cache_tokens,
        max_page_table_width=_page_table_width(max_cache_tokens, page_size),
        # This is a validation ceiling only; physical page count is selected
        # after memory profiling and can exceed max_model_len / page_size when
        # the server has capacity for multiple requests.
        num_cache_pages=_MAX_I32,
        use_cuda_graph=True,
    )
    return dense_mla.plan(caps)


@dataclass
class B12xMLAMetadata(MLACommonMetadata):
    """Common MLA metadata plus the capture-static b12x launch plan."""

    dense_mla_plan: Any | None = None
    dense_mla_scratch: torch.Tensor | None = None
    dense_mla_padded_q: torch.Tensor | None = None
    dense_mla_padded_output: torch.Tensor | None = None
    dense_mla_flat_block_table: torch.Tensor | None = None
    dense_mla_flat_seq_lens: torch.Tensor | None = None
    dense_mla_flat_query_start_loc: torch.Tensor | None = None


class B12xMLAMetadataBuilder(MLACommonMetadataBuilder[B12xMLAMetadata]):
    # The target verifies the accepted token plus the seven DSpark proposals in
    # one causal block.  B12X exposes a single-query decode primitive, so
    # both that block and the draft's non-causal block are flattened below.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM
    # A DSpark verify block is flattened into independent single-token decode
    # rows. Every row sees the same committed prefix, so sibling draft tokens
    # cannot attend to one another.
    supports_non_causal_multi_token_decode: ClassVar[bool] = True
    supports_direct_dcp_kv_gather: ClassVar[bool] = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            B12xMLAMetadata,
            supports_dcp_with_varlen=True,
        )
        try:
            self._dcp_rank = int(get_dcp_group().rank_in_group)
        except AssertionError:
            # Unit tests may construct the builder before distributed init.
            self._dcp_rank = 0
        if self.non_causal_multi_token_decode:
            # Mirror Triton MLA's DSpark policy. The speculative block has
            # 1 + num_speculative_tokens uniform query rows.
            self._init_reorder_batch_threshold(
                1,
                supports_spec_as_decode=True,
                supports_dcp_with_varlen=True,
            )
        max_dense_mla_rows = int(vllm_config.scheduler_config.max_num_seqs) * int(
            self.reorder_batch_threshold
        )
        if max_dense_mla_rows > _MAX_B12X_QUERY_ROWS:
            raise ValueError(
                "B12X_MLA flattened query capacity exceeds its limit: "
                f"rows={max_dense_mla_rows}, limit={_MAX_B12X_QUERY_ROWS}."
            )
        self._max_dense_mla_rows = max_dense_mla_rows
        # The builder receives the final kernel block size, including K3's
        # hybrid-cache alignment (944 in the production launcher). Planning in
        # the layer constructor would incorrectly see the initial size of 16.
        self._effective_heads = self.num_heads * self.dcp_world_size
        self._kernel_heads = _kernel_query_heads(self.num_heads, self.dcp_world_size)
        self._dense_mla_plan = _create_dense_mla_plan(
            vllm_config,
            device,
            page_size=self.page_size,
            num_q_heads=self._kernel_heads,
            max_total_q=self._max_dense_mla_rows,
            dcp_size=self.dcp_world_size,
        )
        self._workspace_specs = self._dense_mla_plan.shapes_and_dtypes()
        if len(self._workspace_specs) != 1:
            raise RuntimeError("B12X_MLA expected exactly one scratch buffer.")
        scratch_shape, scratch_dtype = self._workspace_specs[0]
        # All layers represented by this metadata builder execute serially on
        # the model stream. Give them one stable arena: CUDA graphs retain its
        # address, while sharing avoids keeping an identical arena per layer.
        self._dense_mla_scratch = torch.empty(
            scratch_shape,
            dtype=scratch_dtype,
            device=device,
        )
        # Keep query-gather and output addresses stable across piecewise eager
        # attention replays. This makes graph ownership explicit and avoids
        # allocating per-layer destinations; each layer still builds a fresh
        # caller-scratch-owned B12X binding. All represented layers
        # execute serially on the model stream, so one pair is sufficient.
        max_rows = self._max_dense_mla_rows
        self._dense_mla_padded_q = torch.empty(
            (max_rows, self._kernel_heads, _K3_ABSORBED_HEAD_DIM),
            dtype=_planned_kv_dtype(vllm_config),
            device=device,
        )
        self._dense_mla_padded_output = torch.empty(
            (max_rows, self._kernel_heads, _K3_KV_LORA_RANK),
            dtype=torch.bfloat16,
            device=device,
        )
        self._dense_mla_flat_block_table: torch.Tensor | None = None
        self._dense_mla_flat_seq_lens: torch.Tensor | None = None
        self._dense_mla_flat_query_start_loc: torch.Tensor | None = None
        self._dense_mla_causal_offsets: torch.Tensor | None = None
        self._dense_mla_flat_global_seq_lens: torch.Tensor | None = None
        self._dense_mla_flat_dcp_remainder: torch.Tensor | None = None
        if self.reorder_batch_threshold > 1:
            max_table_width = int(self._dense_mla_plan.caps.max_page_table_width)
            self._dense_mla_flat_block_table = torch.zeros(
                (self._max_dense_mla_rows, max_table_width),
                dtype=torch.int32,
                device=device,
            )
            self._dense_mla_flat_seq_lens = torch.empty(
                self._max_dense_mla_rows,
                dtype=torch.int32,
                device=device,
            )
            self._dense_mla_flat_query_start_loc = torch.arange(
                self._max_dense_mla_rows + 1,
                dtype=torch.int32,
                device=device,
            )
            self._dense_mla_causal_offsets = torch.arange(
                1 - int(self.reorder_batch_threshold),
                1,
                dtype=torch.int32,
                device=device,
            )
            if self.dcp_world_size > 1:
                self._dense_mla_flat_global_seq_lens = torch.empty(
                    self._max_dense_mla_rows,
                    dtype=torch.int32,
                    device=device,
                )
                self._dense_mla_flat_dcp_remainder = torch.empty_like(
                    self._dense_mla_flat_global_seq_lens
                )
        logger.info_once(
            "B12X dense K3 MLA plan: local_heads=%d, effective_heads=%d, "
            "kernel_heads=%d, "
            "page_size=%d, "
            "max_decode_rows=%d, max_cache_tokens=%d, splits=%d, "
            "shared_scratch=%.2f MiB",
            self.num_heads,
            self._effective_heads,
            self._kernel_heads,
            self.page_size,
            self._max_dense_mla_rows,
            _max_dcp_local_cache_tokens(vllm_config, dcp_size=self.dcp_world_size),
            self._dense_mla_plan.num_splits,
            self._dense_mla_scratch.nbytes / (1 << 20),
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> B12xMLAMetadata:
        metadata = cast(
            B12xMLAMetadata,
            super().build(
                common_prefix_len,
                common_attn_metadata,
                fast_build=fast_build,
            ),
        )
        metadata.dense_mla_plan = self._dense_mla_plan
        metadata.dense_mla_scratch = self._dense_mla_scratch
        metadata.dense_mla_padded_q = self._dense_mla_padded_q
        metadata.dense_mla_padded_output = self._dense_mla_padded_output
        flatten_decode = (
            metadata.decode is not None
            and metadata.num_decodes > 0
            and metadata.num_decode_tokens > metadata.num_decodes
        )
        if flatten_decode:
            if metadata.decode is None or metadata.num_decodes <= 0:
                raise ValueError("B12X_MLA flattened metadata requires decode rows.")
            total_q = int(metadata.num_decode_tokens)
            if total_q > self._max_dense_mla_rows:
                raise ValueError(
                    "B12X_MLA query block exceeds its flattened capacity: "
                    f"rows={total_q}, capacity={self._max_dense_mla_rows}."
                )
            if total_q % metadata.num_decodes:
                raise ValueError(
                    "B12X_MLA requires a uniform query block, got "
                    f"tokens={total_q}, requests={metadata.num_decodes}."
                )
            query_len = total_q // metadata.num_decodes
            source_table = metadata.decode.block_table
            source_lens = metadata.decode.seq_lens
            assert self._dense_mla_flat_block_table is not None
            assert self._dense_mla_flat_seq_lens is not None
            assert self._dense_mla_flat_query_start_loc is not None
            flat_table = self._dense_mla_flat_block_table[:total_q]
            # A bounded DSpark cache keeps a position-indexed worker table for
            # the target's complete context, but compacts its resident tail to
            # the beginning before this builder runs. Copy only the page-table
            # prefix the bounded dense-MLA plan can consume. cache_seqlens is
            # shortened by the same compaction, so entries beyond this prefix
            # are unreachable.
            source_width = min(int(source_table.shape[1]), int(flat_table.shape[1]))
            flat_table[:, :source_width].copy_(
                source_table[:, None, :source_width]
                .expand(-1, query_len, -1)
                .reshape(total_q, source_width)
            )
            flat_lens = self._dense_mla_flat_seq_lens[:total_q]
            if metadata.causal:
                # Every flattened row may see the committed prefix and only
                # the verification tokens at or before its own position.
                assert self._dense_mla_causal_offsets is not None
                if self.dcp_world_size > 1:
                    # ``source_lens`` is already local under DCP, so it cannot
                    # be decremented once per *global* verification token.
                    # Build every row from the preserved global final length,
                    # then map it through the same round-robin layout used by
                    # the KV-cache slot mapper.
                    global_source_lens = metadata.decode.dcp_tot_seq_lens
                    if global_source_lens is None:
                        raise RuntimeError(
                            "B12X_MLA DSpark DCP verification requires global "
                            "decode sequence lengths."
                        )
                    assert self._dense_mla_flat_global_seq_lens is not None
                    assert self._dense_mla_flat_dcp_remainder is not None
                    global_flat_lens = self._dense_mla_flat_global_seq_lens[:total_q]
                    torch.add(
                        global_source_lens[:, None],
                        self._dense_mla_causal_offsets[-query_len:],
                        out=global_flat_lens.view(metadata.num_decodes, query_len),
                    )
                    _dcp_local_seq_lens_from_global(
                        flat_lens,
                        self._dense_mla_flat_dcp_remainder[:total_q],
                        global_flat_lens,
                        dcp_size=self.dcp_world_size,
                        dcp_rank=self._dcp_rank,
                        interleave=self.cp_kv_cache_interleave_size,
                    )
                else:
                    # ``source_lens`` includes the complete verification block.
                    torch.add(
                        source_lens[:, None],
                        self._dense_mla_causal_offsets[-query_len:],
                        out=flat_lens.view(metadata.num_decodes, query_len),
                    )
            else:
                flat_lens.copy_(
                    source_lens[:, None].expand(-1, query_len).reshape(total_q)
                )
            metadata.dense_mla_flat_block_table = flat_table
            metadata.dense_mla_flat_seq_lens = flat_lens
            metadata.dense_mla_flat_query_start_loc = (
                self._dense_mla_flat_query_start_loc[: total_q + 1]
            )
        return metadata


class B12xMLABackend(MLACommonBackend):
    """Opt-in dense Kimi K3 MLA backend backed by b12x."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # Keep the literal visible to the generated backend capability table.
        return [576]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3)
        return (0, 1, 2)

    @staticmethod
    def get_name() -> str:
        return "B12X_MLA"

    @staticmethod
    def get_impl_cls() -> type[B12xMLAImpl]:
        return B12xMLAImpl

    @staticmethod
    def get_builder_cls() -> type[B12xMLAMetadataBuilder]:
        return B12xMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 12 and capability.minor in (0, 1)

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        try:
            _load_dense_mla()
        except (ImportError, AttributeError):
            return "B12X_MLA requires a b12x build that provides dense_mla"

        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config
        if model_config is None:
            return None
        hf_text_config = model_config.hf_text_config
        if getattr(hf_text_config, "model_type", None) not in (
            "kimi_linear",
            "k3_dspark",
        ):
            return "B12X_MLA currently supports only Kimi K3 and K3 DSpark"

        dims = (
            getattr(hf_text_config, "kv_lora_rank", None),
            getattr(hf_text_config, "qk_nope_head_dim", None),
            getattr(hf_text_config, "qk_rope_head_dim", None),
            getattr(hf_text_config, "v_head_dim", None),
        )
        required_dims = (
            _K3_KV_LORA_RANK,
            _K3_QK_NOPE_HEAD_DIM,
            _K3_QK_ROPE_HEAD_DIM,
            _K3_V_HEAD_DIM,
        )
        if dims != required_dims:
            return (
                "B12X_MLA requires K3 MLA dimensions "
                "(kv_lora=512, qk_nope=128, qk_rope=64, v=128), "
                f"got {dims}"
            )

        parallel_config = vllm_config.parallel_config
        if parallel_config.prefill_context_parallel_size != 1:
            return "B12X_MLA does not support prefill context parallelism"
        dcp_size = int(parallel_config.decode_context_parallel_size)
        local_heads = model_config.get_num_attention_heads(parallel_config)
        try:
            _kernel_query_heads(local_heads, dcp_size)
        except ValueError as exc:
            return str(exc)
        if vllm_config.scheduler_config.max_num_seqs > _MAX_B12X_QUERY_ROWS:
            return (
                "B12X_MLA max_num_seqs exceeds its 1024-row decode capacity: "
                f"{vllm_config.scheduler_config.max_num_seqs}"
            )
        local_cache_tokens = _max_dcp_local_cache_tokens(vllm_config)
        if local_cache_tokens > _MAX_B12X_CACHE_TOKENS:
            return (
                "B12X_MLA local DCP cache exceeds its 1048576-token capacity: "
                f"{local_cache_tokens}"
            )
        return None

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True


class B12xMLAImpl(MLACommonImpl[B12xMLAMetadata]):
    can_return_lse_for_decode: bool = True
    supports_quant_query_input: bool = True

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
        **mla_args: Any,
    ) -> None:
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
            **mla_args,
        )

        if any(
            feature is not None
            for feature in (alibi_slopes, sliding_window, logits_soft_cap)
        ):
            raise NotImplementedError(
                "B12xMLAImpl does not support alibi, sliding windows, or "
                "logit soft caps."
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("B12xMLAImpl supports decoder attention only.")
        if num_kv_heads != 1:
            raise ValueError(f"B12xMLAImpl requires one KV head, got {num_kv_heads}.")

        actual_dims = (
            head_size,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.qk_head_dim,
            self.v_head_dim,
        )
        required_dims = (
            _K3_ABSORBED_HEAD_DIM,
            _K3_KV_LORA_RANK,
            _K3_QK_NOPE_HEAD_DIM,
            _K3_QK_ROPE_HEAD_DIM,
            _K3_QK_HEAD_DIM,
            _K3_V_HEAD_DIM,
        )
        if actual_dims != required_dims:
            raise ValueError(
                f"B12xMLAImpl received non-K3 MLA dimensions {actual_dims}; "
                f"required {required_dims}."
            )
        vllm_config = get_current_vllm_config()
        dcp_world_size = int(vllm_config.parallel_config.decode_context_parallel_size)
        # KimiK3Attention calls this implementation directly instead of going
        # through MLAAttention.forward(), where the common MLA path normally
        # replaces the sentinel value (-1) with the initialized DCP group size.
        # Keep the configured value here so TP16/DCP8 gathers 6 local heads to
        # the 48-head shape used by the B12X plan.
        self.dcp_world_size = dcp_world_size
        self._effective_heads = num_heads * dcp_world_size
        self._kernel_heads = _kernel_query_heads(num_heads, dcp_world_size)
        if vllm_config.parallel_config.prefill_context_parallel_size != 1:
            raise NotImplementedError(
                "B12xMLAImpl does not support prefill context parallelism."
            )

        self._dense_mla = _load_dense_mla()
        self._dcp_comm_backend = vllm_config.parallel_config.dcp_comm_backend
        self._dcp_max_batch_size = vllm_config.scheduler_config.max_num_batched_tokens
        self._compiled_bindings: set[tuple[object, ...]] = set()
        self._scratch_by_plan: dict[int, torch.Tensor] = {}
        self._padded_io_by_plan: dict[
            tuple[int, torch.dtype, torch.device], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def _borrow_scratch(self, plan: Any, device: torch.device) -> torch.Tensor:
        """Return fallback storage stable for this backend instance.

        Production metadata provides a group-shared arena. Direct backend
        users may not, so retain the per-backend private allocation as a compatibility
        path rather than borrowing vLLM's resizable general workspace.
        """
        specs = plan.shapes_and_dtypes()
        key = id(plan)
        scratch = self._scratch_by_plan.get(key)
        if scratch is None:
            if len(specs) != 1:
                raise RuntimeError("B12X_MLA expected exactly one scratch buffer.")
            shape, dtype = specs[0]
            scratch = torch.empty(shape, dtype=dtype, device=device)
            self._scratch_by_plan[key] = scratch
        return scratch

    def _borrow_padded_io(
        self,
        plan: Any,
        q: torch.Tensor,
        batch: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compatibility storage for direct users lacking production metadata."""
        key = (id(plan), q.dtype, q.device)
        buffers = self._padded_io_by_plan.get(key)
        if buffers is None or int(buffers[0].shape[0]) < batch:
            if q.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "B12X_MLA cannot allocate padded query storage during CUDA "
                    "graph capture; use B12xMLAMetadataBuilder."
                )
            buffers = (
                torch.empty(
                    (batch, self._kernel_heads, _K3_ABSORBED_HEAD_DIM),
                    dtype=q.dtype,
                    device=q.device,
                ),
                torch.empty(
                    (batch, self._kernel_heads, _K3_KV_LORA_RANK),
                    dtype=torch.bfloat16,
                    device=q.device,
                ),
            )
            self._padded_io_by_plan[key] = buffers
        return buffers[0][:batch], buffers[1][:batch]

    def _bind_dense_mla(
        self,
        plan: Any,
        *,
        scratch: torch.Tensor,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        q_scale: torch.Tensor | None,
        kv_scale: torch.Tensor | None,
        active_splits: int,
    ) -> Any:
        """Build a fresh binding from metadata-validated caller scratch."""
        return self._dense_mla.bind(
            plan,
            scratch=scratch,
            q=q,
            kv_cache=kv_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            q_scale=q_scale,
            kv_scale=kv_scale,
            sm_scale=self.scale,
            active_splits=active_splits,
            validate=False,
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: B12xMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if kv_c_and_k_pe_cache.numel() == 0:
            raise ValueError("B12X_MLA received an empty KV cache.")
        if attn_metadata.decode is None:
            raise ValueError("B12X_MLA requires decode metadata.")
        plan = attn_metadata.dense_mla_plan
        if plan is None:
            raise RuntimeError("B12X_MLA metadata is missing its dense MLA plan.")

        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        if not q.is_contiguous():
            q = q.contiguous()

        block_table = attn_metadata.decode.block_table
        seq_lens = attn_metadata.decode.seq_lens
        query_start_loc = attn_metadata.query_start_loc
        flat_block_table = getattr(attn_metadata, "dense_mla_flat_block_table", None)
        if flat_block_table is not None:
            block_table = flat_block_table
            seq_lens = getattr(attn_metadata, "dense_mla_flat_seq_lens", None)
            query_start_loc = getattr(
                attn_metadata, "dense_mla_flat_query_start_loc", None
            )
            if block_table is None or seq_lens is None or query_start_loc is None:
                raise RuntimeError(
                    "B12X_MLA metadata is missing flattened decode rows."
                )
        batch = int(seq_lens.shape[0])
        if int(q.shape[0]) != batch:
            raise ValueError(
                "B12X_MLA's single-token decode path requires one query row per "
                f"request, got {q.shape[0]} rows for {batch} requests."
            )
        if int(q.shape[1]) != self.num_heads:
            raise ValueError(
                f"B12X_MLA expected {self.num_heads} query heads, got {q.shape[1]}."
            )

        dcp_group = None
        if self.dcp_world_size > 1:
            dcp_group = get_dcp_group()
            gathered_q = getattr(attn_metadata, "dense_mla_padded_q", None)
            if gathered_q is not None:
                if int(gathered_q.shape[0]) < batch:
                    raise ValueError(
                        "B12X_MLA shared query buffer is too small: "
                        f"capacity={gathered_q.shape[0]}, batch={batch}."
                    )
                gathered_q = gathered_q[:batch, : self._effective_heads]
            q = dcp_b12x_all_gather_heads(
                q,
                dcp_group,
                max_batch_size=self._dcp_max_batch_size,
                output_head_dim=self.kv_lora_rank,
                out=gathered_q,
            )

        effective_heads = int(q.shape[1])
        if effective_heads != self._effective_heads:
            raise ValueError(
                "B12X_MLA gathered an unexpected query-head count: "
                f"expected {self._effective_heads}, got {effective_heads}."
            )
        padded_q = getattr(attn_metadata, "dense_mla_padded_q", None)
        padded_output = getattr(attn_metadata, "dense_mla_padded_output", None)
        if self._kernel_heads != effective_heads:
            if padded_q is None or padded_output is None:
                padded_q, padded_output = self._borrow_padded_io(plan, q, batch)
            else:
                if int(padded_q.shape[0]) < batch:
                    raise ValueError(
                        "B12X_MLA shared padded query buffer is too small: "
                        f"capacity={padded_q.shape[0]}, batch={batch}."
                    )
                padded_q = padded_q[:batch]
                padded_output = padded_output[:batch]
            if padded_q.dtype != q.dtype:
                raise TypeError(
                    "B12X_MLA padded query dtype does not match the live query: "
                    f"buffer={padded_q.dtype}, query={q.dtype}."
                )
            padded_q[:, :effective_heads].copy_(q)
            padded_q[:, effective_heads:].zero_()
            q = padded_q
            output = padded_output
        else:
            if padded_output is None:
                _, padded_output = self._borrow_padded_io(plan, q, batch)
            elif int(padded_output.shape[0]) < batch:
                raise ValueError(
                    "B12X_MLA shared output buffer is too small: "
                    f"capacity={padded_output.shape[0]}, batch={batch}."
                )
            output = padded_output[:batch]
        scratch = getattr(attn_metadata, "dense_mla_scratch", None)
        if scratch is None:
            # Compatibility fallback for direct backend users and metadata
            # builders without shared scratch. K3 serving metadata supplies
            # the capture-stable arena allocated above.
            scratch = self._borrow_scratch(plan, q.device)
        quantized = q.dtype == torch.float8_e4m3fn
        # A conventional CUDA graph fixes kernel grid dimensions and scalar
        # launch arguments at capture time.  Breakable PIECEWISE execution
        # calls attention outside the captured segments, so it reaches the
        # adaptive branch below on every replay.  If another graph mode ever
        # captures this backend directly, retain the full plan for correctness.
        if q.is_cuda and torch.cuda.is_current_stream_capturing():
            active_splits = int(plan.num_splits)
        else:
            active_splits = _active_dense_mla_splits(
                plan,
                getattr(attn_metadata, "max_seq_len", None),
            )
        cu_seqlens_q = query_start_loc[: batch + 1]
        binding = self._bind_dense_mla(
            plan,
            scratch=scratch,
            q=q,
            kv_cache=kv_c_and_k_pe_cache,
            output=output,
            page_table=block_table,
            cache_seqlens=seq_lens,
            cu_seqlens_q=cu_seqlens_q,
            q_scale=layer._q_scale if quantized else None,
            kv_scale=layer._k_scale if quantized else None,
            active_splits=active_splits,
        )

        compile_key = (
            id(plan),
            q.dtype,
            tuple(q.stride()),
            tuple(kv_c_and_k_pe_cache.stride()),
            tuple(output.stride()),
        )
        if compile_key not in self._compiled_bindings:
            if q.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "B12X_MLA encountered an uncompiled layout during CUDA graph "
                    "capture; the eager graph warmup did not exercise this cache "
                    "layout."
                )
            self._dense_mla.compile(binding=binding)
            self._compiled_bindings.add(compile_key)

        if is_b12x_compile_only_warmup():
            # The binding above carries the exact FP8/BF16 strides retained by
            # capture. Compilation is sufficient here: launching the complete
            # K3 warmup forward against capture-prepared workspaces is unsafe.
            local_output = output[:, : self.num_heads]
            local_output.zero_()
            return local_output, None

        output, lse = self._dense_mla.run(binding=binding)
        if os.environ.get("VLLM_KIMI_DEBUG_FINITE") == "1":
            torch.cuda.synchronize()
            # An empty DCP shard is represented by LSE=-inf. B12X is allowed
            # to leave that shard's local output undefined (often NaN): the
            # DCP correction kernel assigns it zero softmax weight and writes
            # zero before reduce-scatter. Reject non-finite output only for an
            # active row with finite LSE; the post-reduction guard below still
            # requires the merged output to be fully finite.
            output_rows_finite = torch.isfinite(output).all(dim=-1)
            empty_rows = torch.isneginf(lse)
            output_ok = bool((output_rows_finite | empty_rows).all().item())
            lse_ok = bool((~torch.isnan(lse) & ~torch.isposinf(lse)).all().item())
            if not output_ok or not lse_ok:
                raise RuntimeError(
                    "B12X_MLA produced invalid active local output/LSE: "
                    f"active_output_finite={output_ok}, lse_valid={lse_ok}"
                )
        if dcp_group is None:
            return output[:, : self.num_heads], lse[:, : self.num_heads]

        if self._dcp_comm_backend == "a2a":
            reduced = dcp_a2a_lse_reduce(
                output,
                lse,
                dcp_group,
                is_lse_base_on_e=True,
                use_b12x=True,
                b12x_max_batch_size=self._dcp_max_batch_size,
                b12x_query_head_dim=_K3_ABSORBED_HEAD_DIM,
            )
        else:
            reduced = cp_lse_ag_out_rs(
                output,
                lse,
                dcp_group,
                is_lse_base_on_e=True,
            )
        if os.environ.get("VLLM_KIMI_DEBUG_FINITE") == "1":
            torch.cuda.synchronize()
            if not bool(torch.isfinite(reduced).all().item()):
                raise RuntimeError("B12X_MLA DCP reduction produced nonfinite output")
        return reduced, None


__all__ = [
    "B12xMLABackend",
    "B12xMLAImpl",
    "B12xMLAMetadata",
    "B12xMLAMetadataBuilder",
]
