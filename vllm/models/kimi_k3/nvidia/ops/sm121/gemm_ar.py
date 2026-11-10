# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GEMM-AR on SM12x: a row-parallel projection fused with its all-reduce.

The SM100 GEMM-AR (``kernels/linear/cute_dsl/gemm_rs_ar``) is a TMEM GEMM that
reduces through NVLink multicast memory.  A Spark rank has neither; its
counterpart is ``roce_gemm.cuh``: the GEMM runs in the same launch as a
RoCEnante two-shot all-reduce and writes its partial straight into the
reduce-scatter windows of the send slot.

The regimes differ.  The SM100 kernel serves batches of 128 rows and more
(prefill).  On RoCE a rank's partial of such a batch does not fit one send slot,
and measured on 16 Sparks the collective is then bound by the wire: a
[4096, 7168] BF16 all-reduce takes 15.3 ms as 32 chunked two-shot launches
against 8.8 ms on NCCL (1.9 ms against 1.8 ms at 512 rows, 0.48 ms against
0.78 ms at 128 rows, which the adapter's two-shot already takes).  The fused
launch here therefore covers decode batches (up to ``GEMM_MAX_ROWS`` rows), and
larger batches keep the projection's own forward.  At decode sizes it is exact
but not faster than cuBLAS followed by the two-shot kernel (52.6 us against
50.5 us at 6 rows, 41.2 us against 40.8 us at one): a GEMM over the 11 MB weight
wants every SM, the collective wants one 8-block cluster, and spanning both in
one launch costs about as much in cluster launches and the grid barrier as the
staging copy and the second launch it saves.  It is opt-in for that reason.

:class:`Sm121GemmAr` has the interface of ``GemmRsAr`` (``can_run``,
``should_run``, ``apply``) so the projection call sites dispatch on it
unchanged; :func:`get_gemm_rs_ar` returns whichever backend was initialised.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from .roce_fused import GEMM_K_STEP, GEMM_MAX_ROWS

logger = init_logger(__name__)

_INSTANCE: "Sm121GemmAr | None" = None


def tp_fused_collectives():
    """The TP group's RoCE fused collectives, or None (looked up per call: a
    transport sleep/wake cycle rebuilds the adapter that owns them)."""
    from vllm.distributed.parallel_state import get_tp_group

    adapter = getattr(get_tp_group().device_communicator, "b12x_ar_comm", None)
    if adapter is None or adapter.disabled:
        return None
    return getattr(adapter, "fused", None)


class Sm121GemmAr:
    """Decode-batch GEMM + all-reduce for unquantized BF16 projections."""

    def __init__(self, hidden_size: int) -> None:
        self.N = hidden_size

    def can_run(self, linear: LinearBase) -> bool:
        # Called before weight loading: shapes and dtypes only.
        if not isinstance(linear.quant_method, UnquantizedLinearMethod):
            return False
        weight = linear.weight
        return (
            weight.dtype == torch.bfloat16
            and weight.ndim == 2
            and weight.shape[0] == self.N
            and weight.shape[1] % GEMM_K_STEP == 0
            and getattr(linear, "bias", None) is None
        )

    def warn_incompatible_projection(self) -> None:
        logger.warning_once(
            "Some projections are incompatible with the SM12x GEMM-AR; using "
            "the unfused path instead.",
            scope="global",
        )

    def should_run(self, x: torch.Tensor) -> bool:
        # Rank-invariant: the batch shape and the (voted) fused collectives.
        if x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[0] > GEMM_MAX_ROWS:
            return False
        return tp_fused_collectives() is not None

    def apply(self, x: torch.Tensor, linear: LinearBase) -> torch.Tensor:
        fused = tp_fused_collectives()
        assert fused is not None
        weight = linear.weight
        if not fused.can_gemm_all_reduce(x, weight):
            # Same on every rank (shapes only): the projection's own forward.
            return linear(x)[0]
        return fused.gemm_all_reduce(x, weight)


def init_sm121_gemm_ar(hidden_size: int) -> bool:
    global _INSTANCE
    _INSTANCE = Sm121GemmAr(hidden_size)
    logger.info_once(
        "Kimi-K3 GEMM-AR: SM12x RoCE kernel for batches of up to %d rows "
        "(VLLM_KIMI_K3_SM121_GEMM_AR).",
        GEMM_MAX_ROWS,
    )
    return True


def get_gemm_rs_ar():
    """The GEMM-RS/AR backend this worker initialised."""
    if _INSTANCE is not None:
        return _INSTANCE
    from vllm.model_executor.kernels.linear.cute_dsl.gemm_rs_ar import (
        get_gemm_rs_ar as get_sm100_gemm_rs_ar,
    )

    return get_sm100_gemm_rs_ar()
