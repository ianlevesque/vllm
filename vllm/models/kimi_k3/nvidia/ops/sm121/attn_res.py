# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AttnRes forward for SM12x (see attn_res.cuh)."""

import torch

from ._ext import load_extension
from ._scratch import ATTN_RES_MAX_ROWS, AttnResScratch

MAX_BLOCKS = 8
PACK_ELEMS = 8  # elements of one 16-byte BF16 pack
HIDDEN = 7168  # one 7-block cluster of 128 threads, one 16-byte pack each
MAX_ROWS = ATTN_RES_MAX_ROWS


def _vector_ok(tensor: torch.Tensor, hidden: int) -> bool:
    return (
        tensor.dtype == torch.bfloat16
        and tensor.shape[-1] == hidden
        and tensor.stride(-1) == 1
        and tensor.data_ptr() % 16 == 0
    )


def is_supported(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
    block_write_idx: int,
    output: torch.Tensor | None,
) -> bool:
    """Whether these operands satisfy the kernel's layout contract.

    Every vector is read and written as aligned 16-byte packs, so row strides
    must be multiples of eight elements and base pointers 16-byte aligned.
    """
    hidden = prefix.shape[-1]
    if (
        prefix.ndim != 2
        or not 0 < prefix.shape[0] <= MAX_ROWS
        or hidden != HIDDEN
        or not 0 <= num_blocks <= MAX_BLOCKS
        or blocks.ndim != 3
        or num_blocks > blocks.shape[1]
        or not -1 <= block_write_idx < blocks.shape[1]
    ):
        return False
    operands = [prefix, blocks, norm_weight, qk_weight]
    if delta is not None:
        operands.append(delta)
    if output_norm_weight is not None:
        operands.append(output_norm_weight)
    if output is not None:
        operands.append(output)
    if not all(_vector_ok(t, hidden) for t in operands):
        return False
    strides = [prefix.stride(0), blocks.stride(0), blocks.stride(1)]
    if delta is not None:
        strides.append(delta.stride(0))
    if output is not None:
        strides.append(output.stride(0))
    return all(stride % PACK_ELEMS == 0 for stride in strides)


def attn_res(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
    output: torch.Tensor | None = None,
    *,
    scratch: AttnResScratch,
) -> torch.Tensor:
    """Same contract as ``ops.attn_res.attn_res`` plus this call site's scratch.

    Callers check ``is_supported`` first.
    """
    rows, hidden = prefix.shape
    if output is None:
        output = prefix.new_empty(prefix.shape)
    load_extension().attn_res(
        prefix,
        prefix.stride(0),
        delta,
        0 if delta is None else delta.stride(0),
        blocks,
        blocks.stride(0),
        blocks.stride(1),
        norm_weight,
        qk_weight,
        output_norm_weight,
        output,
        output.stride(0),
        scratch.partial,
        rows,
        num_blocks,
        block_write_idx,
        float(eps),
        float(output_norm_eps),
    )
    return output
