# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stand-ins for the tokenspeed_kernel modules the vendored TokenSpeed
kernels import: the registry decorators become no-ops (vLLM calls the
portable kernels directly), the mailbox fold path is never taken, and
`_prefetch_weight_rows` is copied from tokenspeed_kernel/ops/gemm/
triton_skinny.py (MIT, LightSeek Foundation) for the SM10x-only scaled
GEMM path that stays unused here."""

from dataclasses import dataclass
from typing import Any

from triton.tools.tensor_descriptor import TensorDescriptor

from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as libdevice

__all__ = [
    "ArchVersion",
    "CapabilityRequirement",
    "MAILBOX_ROWS",
    "MailboxTarget",
    "Priority",
    "TensorDescriptor",
    "dense_tensor_format",
    "format_signature",
    "format_signatures",
    "libdevice",
    "pdl_enabled",
    "register_kernel",
    "tl",
    "triton",
    "_prefetch_weight_rows",
]

# No mailbox target is ever passed (fold_into is None or a tensor).
MAILBOX_ROWS = 0


class MailboxTarget:
    """Never instantiated: `isinstance(x, MailboxTarget)` is always False."""


@dataclass(frozen=True)
class ArchVersion:
    major: int
    minor: int


@dataclass(frozen=True)
class CapabilityRequirement:
    vendors: Any = None
    min_arch_version: Any = None
    max_arch_version: Any = None


class Priority:
    PORTABLE = 0
    PERFORMANT = 1


def register_kernel(*_args: Any, **_kwargs: Any):
    def decorate(fn):
        return fn

    return decorate


def format_signature(**_kwargs: Any) -> None:
    return None


def format_signatures(*_args: Any, **_kwargs: Any) -> frozenset:
    return frozenset()


def dense_tensor_format(*_args: Any, **_kwargs: Any) -> None:
    return None


def pdl_enabled() -> bool:
    # Programmatic dependent launch stays off for the portable path.
    return False


@triton.jit
def _prefetch_weight_rows(W, n0, k0, N, stride_w, BW: tl.constexpr, BYTES):
    """One bulk L2 prefetch per weight row of the slice."""
    rows = n0 + tl.arange(0, BW)
    address = (W + tl.minimum(rows, N - 1).to(tl.int64) * stride_w + k0).to(
        tl.pointer_type(tl.int8)
    )
    tl.inline_asm_elementwise(
        """
        {
        cp.async.bulk.prefetch.L2.global [$1], $2;
        mov.u32 $0, 0;
        }
        """,
        "=r,l,r",
        [address, tl.full((BW,), BYTES, tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
