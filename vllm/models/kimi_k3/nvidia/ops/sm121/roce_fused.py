# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RoCEnante fused collectives for SM12x (see roce_fused.cuh, roce_gemm.cuh).

Thin launch layer over one ``b12x.comm.roce`` runtime: it checks that the
runtime's proxy posts windowed ops, validates operand layout (the kernels read
and write aligned 16-byte packs), and launches inside the runtime's
``fused_launch`` admission so the kernels are ordered with, and fail-stop
like, the runtime's own collectives.
"""

import torch

from ._ext import load_extension

PACK_BYTES = 16
PACK_ELEMS = 8  # BF16 values per pack
TAIL_MAX_ROWS = 9
TAIL_SCRATCH_FLOATS = 128
WINDOW_FEATURE = 1
MAX_WINDOW_PACKS = 0x3FFF
MAX_COMMON_PACKS = 0xFFFF
# Fused small-M GEMM (roce_gemm.cuh): one tensor-core MMA covers up to 16 rows
# of x, a warp step covers 32 input features, and a block holds x as BF16
# [rows, K] in shared memory next to 8 KB of accumulators, so the device's
# per-block limit (99 KB on GB10) bounds rows * K.
GEMM_MAX_ROWS = 16
GEMM_K_STEP = 32
GEMM_BOUNCE_BYTES = 512 * 16
GEMM_MODE_TWO_SHOT = 0
GEMM_MODE_GATHER = 1

_HANDLE_ORDER = (
    "recv_base",
    "flag_base",
    "send_base",
    "ctrl_base",
    "slot_bytes",
    "epoch_address",
    "poison_address",
    "spin_limit",
    "world_size",
    "rank",
    "hca_count",
    "slots",
    "flag_stride",
)


def unsupported_reason(runtime) -> str | None:
    """Why ``runtime`` cannot run the fused collectives, or None when it can."""
    if runtime is None:
        return "no RoCEnante runtime"
    if not hasattr(runtime, "fused_handles") or not hasattr(runtime, "fused_launch"):
        return "b12x.comm.roce lacks the integrator hooks (fused_handles)"
    if not getattr(runtime, "supports_windows", False):
        return "the RoCE proxy does not post windowed ops"
    if runtime.world_size * len(runtime.hca_names) > 32:
        return "more stripe flags than one warp can poll"
    return None


def two_shot_geometry_ok(rows: int, columns: int, world: int, slot_bytes: int) -> bool:
    """Whether a [rows, columns] BF16 two-shot all-reduce fits the transport.

    Every rank owns ``columns / world`` whole 16-byte packs per row, a window
    (one rank's columns of every row) fits the doorbell word, and all windows
    fit one send slot.
    """
    if rows <= 0 or columns <= 0 or columns % (world * PACK_ELEMS) != 0:
        return False
    window_packs = rows * (columns // world) // PACK_ELEMS
    return (
        window_packs <= MAX_WINDOW_PACKS
        and world * window_packs * PACK_BYTES <= slot_bytes
    )


def moe_tail_geometry_ok(
    rows: int, latent_dim: int, columns: int, world: int, slot_bytes: int
) -> bool:
    """Whether the latent-MoE tail collective covers these shapes.

    A latent row must be whole warps of packs (the RMSNorm reduction is per
    warp), both widths must split into whole packs per rank, the rows must fit
    the cluster's 128 warps, and the per-peer window must fit the transport.
    """
    if (
        not 0 < rows <= TAIL_MAX_ROWS
        or latent_dim <= 0
        or columns <= 0
        or latent_dim % (32 * PACK_ELEMS) != 0
        or latent_dim % (world * PACK_ELEMS) != 0
        or columns % (world * PACK_ELEMS) != 0
        or rows * (latent_dim // (32 * PACK_ELEMS)) > 128
    ):
        return False
    window_packs = rows * ((latent_dim + columns) // world) // PACK_ELEMS
    return (
        window_packs <= MAX_WINDOW_PACKS
        and world * window_packs * PACK_BYTES <= slot_bytes
    )


def gather_geometry_ok(row_bytes: tuple[int, ...], rows: int, slot_bytes: int) -> bool:
    """Whether 1-2 tensors of ``rows`` rows gather in one round."""
    if not 1 <= len(row_bytes) <= 2 or rows <= 0:
        return False
    if any(b <= 0 or b % PACK_BYTES != 0 for b in row_bytes):
        return False
    total = rows * sum(row_bytes)
    return total <= min(slot_bytes, MAX_COMMON_PACKS * PACK_BYTES)


def gemm_geometry_ok(rows: int, k: int, columns: int, shared_bytes: int) -> bool:
    """Whether the fused GEMM kernel computes ``[rows, k] @ [columns, k]^T``.

    Mirrors the launch checks of ``roce_gemm``: the rows fit one MMA, a weight
    row is whole warp steps, the output is whole packs, and x fits
    ``shared_bytes`` of shared memory per block.
    """
    return (
        0 < rows <= GEMM_MAX_ROWS
        and k >= GEMM_K_STEP
        and k % GEMM_K_STEP == 0
        and columns > 0
        and columns % PACK_ELEMS == 0
        and rows * k * 2 + GEMM_BOUNCE_BYTES <= shared_bytes
    )


def _bf16_matrix(tensor: torch.Tensor) -> bool:
    return (
        tensor.dtype == torch.bfloat16
        and tensor.ndim == 2
        and tensor.is_contiguous()
        and tensor.data_ptr() % PACK_BYTES == 0
    )


class RoceFusedCollectives:
    """Fused collectives of one tensor-parallel group on one RoCE runtime."""

    def __init__(self, runtime, device: torch.device) -> None:
        reason = unsupported_reason(runtime)
        if reason is not None:
            raise RuntimeError(f"RoCE fused collectives unavailable: {reason}")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "RoCE fused collectives must be built before CUDA graph capture"
            )
        self.runtime = runtime
        self.device = device
        handles = runtime.fused_handles()
        self.world = int(handles["world_size"])
        self.rank = int(handles["rank"])
        self.slot_bytes = int(handles["slot_bytes"])
        self._handles = [int(handles[name]) for name in _HANDLE_ORDER]
        self._tail_scratch = torch.zeros(
            TAIL_SCRATCH_FLOATS, dtype=torch.float32, device=device
        )
        # Grid barrier word of the fused GEMM kernels (zero between launches).
        self._gemm_barrier = torch.zeros(1, dtype=torch.int32, device=device)
        self._ext = load_extension()
        self.gemm_shared_bytes = int(self._ext.roce_gemm_reserve())

    # -- two-shot all-reduce ------------------------------------------------

    def can_two_shot_all_reduce(self, inp: torch.Tensor) -> bool:
        """Rank-invariant eligibility: dtype, shape and size only."""
        if inp.dtype != torch.bfloat16 or inp.ndim != 2 or not inp.is_contiguous():
            return False
        rows, columns = inp.shape
        return two_shot_geometry_ok(rows, columns, self.world, self.slot_bytes)

    def two_shot_all_reduce(
        self, inp: torch.Tensor, *, stream: torch.cuda.Stream | None = None
    ) -> torch.Tensor:
        """Sum ``inp`` ([rows, N] BF16) over the ranks.

        Round 1 sends each peer only the columns it owns (a reduce-scatter);
        each owner sums its columns in rank order in FP32 and rounds once;
        round 2 all-gathers the reduced shards.  Every rank returns identical
        bits.  A rank moves 2 * N / world columns per peer instead of N.
        """
        assert self.can_two_shot_all_reduce(inp)
        with self.runtime.fused_launch(stream):
            src = inp
            if inp.data_ptr() % PACK_BYTES != 0:
                src = inp.clone()
            out = torch.empty_like(src)
            assert out.data_ptr() % PACK_BYTES == 0
            self._ext.roce_two_shot_all_reduce(
                self._handles, src, out, src.shape[0], src.shape[1]
            )
        return out

    # -- latent-MoE tail collective -------------------------------------------

    def can_moe_tail_collective(
        self, latent: torch.Tensor, shared: torch.Tensor
    ) -> bool:
        if not (_bf16_matrix(latent) and _bf16_matrix(shared)):
            return False
        rows, latent_dim = latent.shape
        return shared.shape[0] == rows and moe_tail_geometry_ok(
            rows, latent_dim, shared.shape[1], self.world, self.slot_bytes
        )

    def moe_tail_collective(
        self,
        latent: torch.Tensor,
        shared: torch.Tensor,
        norm_weight: torch.Tensor | None,
        eps: float,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The latent-MoE tail's collective: two small rounds in one launch.

        Round 1 reduce-scatters both partials, so a rank sends a peer only the
        latent and shared columns that peer owns; round 2 all-gathers the
        reduced latent, to which the RMSNorm is applied on arrival.

        Args:
            latent: ``[rows, D]`` this rank's routed-expert latent partial.
            shared: ``[rows, N]`` this rank's shared-expert partial.
            norm_weight: ``[D]`` RMSNorm weight applied to the reduced latent
                (None returns the plain sum).
            eps: RMSNorm epsilon.
            stream: launch stream (default: the current stream).

        Returns:
            ``(latent_out, shared_shard)``: the normalised latent sum
            ``[rows, D]`` and the sum over ranks of the ``N / world`` shared
            columns this rank owns ``[rows, N / world]``.

        """
        assert self.can_moe_tail_collective(latent, shared)
        rows, latent_dim = latent.shape
        columns = shared.shape[1]
        if norm_weight is not None:
            assert norm_weight.dtype == torch.bfloat16
            assert norm_weight.shape == (latent_dim,) and norm_weight.is_contiguous()
            assert norm_weight.data_ptr() % PACK_BYTES == 0
        with self.runtime.fused_launch(stream):
            latent_out = torch.empty_like(latent)
            shared_out = torch.empty(
                (rows, columns // self.world), dtype=shared.dtype, device=shared.device
            )
            assert latent_out.data_ptr() % PACK_BYTES == 0
            assert shared_out.data_ptr() % PACK_BYTES == 0
            self._ext.roce_moe_tail_collective(
                self._handles,
                latent,
                shared,
                norm_weight,
                latent_out,
                shared_out,
                self._tail_scratch,
                rows,
                latent_dim,
                columns,
                float(eps),
            )
        return latent_out, shared_out

    # -- fused small-M GEMM ------------------------------------------------------

    def _gemm_operands_ok(self, x: torch.Tensor, weight: torch.Tensor) -> bool:
        """Rank-invariant: dtypes and shapes only (alignment is asserted)."""
        return (
            x.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and x.ndim == 2
            and weight.ndim == 2
            and weight.shape[1] == x.shape[1]
            and weight.is_contiguous()
            and gemm_geometry_ok(
                x.shape[0], x.shape[1], weight.shape[0], self.gemm_shared_bytes
            )
        )

    def can_gemm_all_reduce(self, x: torch.Tensor, weight: torch.Tensor) -> bool:
        return self._gemm_operands_ok(x, weight) and two_shot_geometry_ok(
            x.shape[0], weight.shape[0], self.world, self.slot_bytes
        )

    def gemm_all_reduce(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """``sum over ranks of x @ weight.T`` in one launch (GEMM-AR).

        ``x`` is ``[rows, K]`` and ``weight`` this rank's ``[N, K]`` shard of a
        row-parallel projection.  The partial product is rounded to BF16
        straight into the reduce-scatter windows of the send slot; the rounds
        and the result are those of :meth:`two_shot_all_reduce` applied to
        that partial.
        """
        assert self.can_gemm_all_reduce(x, weight)
        return self._gemm(x, weight, None, weight.shape[0], GEMM_MODE_TWO_SHOT, stream)

    def can_gemm_add_all_gather(
        self, x: torch.Tensor, weight: torch.Tensor, addend: torch.Tensor | None
    ) -> bool:
        if not self._gemm_operands_ok(x, weight):
            return False
        rows, columns = x.shape[0], weight.shape[0]
        if addend is not None and (
            addend.dtype != torch.bfloat16 or addend.shape != (rows, columns)
        ):
            return False
        return gather_geometry_ok((columns * 2,), rows, self.slot_bytes)

    def gemm_add_all_gather(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        addend: torch.Tensor | None = None,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """All-gather of every rank's ``addend + x @ weight.T`` in one launch.

        ``weight`` is this rank's ``[N / world, K]`` shard of a column-parallel
        projection (the latent-MoE tail's up-projection) and ``addend`` its
        ``[rows, N / world]`` shard of the sum it joins.  The shard is rounded
        to BF16 once, straight into the send slot as the payload of the gather
        round.  Returns ``[rows, N]``.
        """
        assert self.can_gemm_add_all_gather(x, weight, addend)
        columns = weight.shape[0] * self.world
        return self._gemm(x, weight, addend, columns, GEMM_MODE_GATHER, stream)

    def _gemm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        addend: torch.Tensor | None,
        out_columns: int,
        mode: int,
        stream: torch.cuda.Stream | None,
    ) -> torch.Tensor:
        assert weight.data_ptr() % PACK_BYTES == 0, "weight must be 16-byte aligned"
        with self.runtime.fused_launch(stream):
            if not x.is_contiguous() or x.data_ptr() % PACK_BYTES != 0:
                x = x.clone(memory_format=torch.contiguous_format)
            if addend is not None and (
                not addend.is_contiguous() or addend.data_ptr() % PACK_BYTES != 0
            ):
                addend = addend.clone(memory_format=torch.contiguous_format)
            out = torch.empty(
                (x.shape[0], out_columns), dtype=torch.bfloat16, device=x.device
            )
            assert out.data_ptr() % PACK_BYTES == 0
            self._ext.roce_gemm(
                self._handles, x, weight, addend, out, self._gemm_barrier, mode
            )
        return out

    # -- all-gather -------------------------------------------------------------

    def can_all_gather(self, *tensors: torch.Tensor) -> bool:
        """Rank-invariant eligibility of a last-dim all-gather of 1-2 tensors."""
        if not 1 <= len(tensors) <= 2:
            return False
        rows = tensors[0].shape[0] if tensors[0].ndim == 2 else 0
        for tensor in tensors:
            if (
                tensor.ndim != 2
                or tensor.shape[0] != rows
                or not tensor.is_cuda
                or not tensor.is_contiguous()
                or tensor.is_complex()
                or tensor.dtype == torch.bool
            ):
                return False
        return gather_geometry_ok(
            tuple(t.shape[1] * t.element_size() for t in tensors), rows, self.slot_bytes
        )

    def all_gather(
        self,
        first: torch.Tensor,
        second: torch.Tensor | None = None,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Concatenate every rank's ``[rows, w]`` tensor(s) along the last dim.

        Two tensors (of any dtypes) share one round: one doorbell, one RDMA
        write per peer and HCA, one wait.
        """
        tensors = (first,) if second is None else (first, second)
        assert self.can_all_gather(*tensors)
        with self.runtime.fused_launch(stream):
            sources = [
                t if t.data_ptr() % PACK_BYTES == 0 else t.clone() for t in tensors
            ]
            outputs = [
                torch.empty(
                    (t.shape[0], t.shape[1] * self.world),
                    dtype=t.dtype,
                    device=t.device,
                )
                for t in sources
            ]
            packs = [t.shape[1] * t.element_size() // PACK_BYTES for t in sources]
            self._ext.roce_all_gather2(
                self._handles,
                sources[0],
                sources[1] if second is not None else None,
                outputs[0],
                outputs[1] if second is not None else None,
                sources[0].shape[0],
                packs[0],
                packs[1] if second is not None else 0,
            )
        return outputs[0] if second is None else (outputs[0], outputs[1])
