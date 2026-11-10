# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused KDA decode / speculative decode for SM12x (see kda_decode.cuh)."""

import torch

from ._ext import load_extension
from ._scratch import KDA_MAX_GROUPS

HEAD_DIM = 128
CONV_WIDTH = 4
MAX_ROWS_PER_SEQUENCE = 8


def fits(num_seqs: int, num_heads: int) -> bool:
    """Whether one launch can hold this many (sequence, head) groups.

    Each group is a cluster of eight blocks that meets at a barrier, so a
    launch is kept small enough for every cluster to be resident at once.
    """
    return 0 < num_seqs * num_heads <= KDA_MAX_GROUPS


def is_supported(
    head_dim: int,
    conv_width: int,
    num_spec: int,
    input_dtype: torch.dtype,
    conv_state_dtype: torch.dtype,
    recurrent_state_dtype: torch.dtype,
    lower_bound: float | None,
    use_recoverssm: bool,
) -> bool:
    """Whether the SM12x fused kernel implements this KDA configuration.

    Unlike the SM100 kernels it replaces, the local head count is unrestricted
    (TP16 leaves six heads per rank) and one kernel serves both the plain and
    the speculative step.
    """
    return (
        head_dim == HEAD_DIM
        and conv_width == CONV_WIDTH
        and 0 <= num_spec < MAX_ROWS_PER_SEQUENCE
        and input_dtype == torch.bfloat16
        and conv_state_dtype == torch.bfloat16
        and recurrent_state_dtype == torch.float32
        and lower_bound is not None
        and lower_bound < 0
        and not use_recoverssm
    )


def kda_fused_decode(
    *,
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    lower_bound: float,
    norm_eps: float,
    output: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convolution, recurrent KDA and gated RMSNorm in one launch.

    Args:
        x: ``[rows, >= 3 * H * 128]`` projected q|k|v rows (BF16, head-major).
        conv_weight: ``[3 * H * 128, 4]`` FP32 taps.
        conv_state: ``[slots, 3 * H * 128, taps]`` BF16 view (any strides).
        raw_gate: ``[1, rows, H, 128]`` raw decay-gate logits.
        raw_beta: ``[1, rows, H]`` raw beta logits.
        A_log: ``[H]`` FP32.
        dt_bias: ``[H * 128]`` FP32.
        state: ``[slots, H, 128, 128]`` FP32 recurrent state.
        state_indices: ``[N]`` (plain decode) or ``[N, T]`` (speculative)
            INT32 cache slots; column 0 also addresses the convolution state.
        output_gate: ``[rows, H, 128]`` output-gate logits.
        norm_weight: ``[128]`` FP32 output-norm weight.
        lower_bound: bounded-gate constant (negative).
        norm_eps: output-norm epsilon.
        output: ``[1, rows, H, 128]`` BF16 destination.
        num_accepted_tokens: ``[N]`` INT32, speculative only.
        query_start_loc: ``[N + 1]`` INT32 row offsets; omitted means one row
            per sequence.

    Returns:
        ``output``.

    """
    rows = output.shape[1]
    num_heads = A_log.shape[0]
    local = num_heads * HEAD_DIM
    assert x.dtype == raw_gate.dtype == raw_beta.dtype == torch.bfloat16
    assert output_gate.dtype == output.dtype == conv_state.dtype == torch.bfloat16
    assert conv_weight.dtype == state.dtype == norm_weight.dtype == torch.float32
    assert A_log.dtype == dt_bias.dtype == torch.float32
    assert state_indices.dtype == torch.int32
    assert x.ndim == 2 and x.shape[1] >= 3 * local and x.stride(1) == 1
    assert conv_weight.shape == (3 * local, CONV_WIDTH) and conv_weight.is_contiguous()
    assert conv_state.ndim == 3 and conv_state.shape[1] >= 3 * local
    assert raw_gate.shape == (1, rows, num_heads, HEAD_DIM)
    assert raw_gate.stride()[2:] == (HEAD_DIM, 1)
    assert raw_beta.shape == (1, rows, num_heads) and raw_beta.stride(2) == 1
    assert state.shape[1:] == (num_heads, HEAD_DIM, HEAD_DIM)
    assert state.stride()[1:] == (HEAD_DIM * HEAD_DIM, HEAD_DIM, 1)
    assert state.data_ptr() % 16 == 0 and state.stride(0) % 4 == 0
    assert output_gate.shape == (rows, num_heads, HEAD_DIM)
    assert output_gate.stride()[1:] == (HEAD_DIM, 1)
    assert output.shape == (1, rows, num_heads, HEAD_DIM)
    assert output.stride()[2:] == (HEAD_DIM, 1)
    assert norm_weight.shape == (HEAD_DIM,) and norm_weight.is_contiguous()
    assert A_log.is_contiguous() and dt_bias.is_contiguous()
    assert dt_bias.numel() == local

    if state_indices.ndim == 2:
        assert state_indices.stride(1) == 1
        num_seqs = (
            query_start_loc.numel() - 1
            if query_start_loc is not None
            else state_indices.shape[0]
        )
        assert conv_state.shape[2] >= state_indices.shape[1] + CONV_WIDTH - 2
    else:
        assert state_indices.ndim == 1
        assert num_accepted_tokens is None
        num_seqs = query_start_loc.numel() - 1 if query_start_loc is not None else rows
        assert conv_state.shape[2] >= CONV_WIDTH - 1
    if num_accepted_tokens is not None:
        assert num_accepted_tokens.dtype == torch.int32
        assert num_accepted_tokens.is_contiguous()
    if query_start_loc is not None:
        assert query_start_loc.dtype == torch.int32
        assert query_start_loc.is_contiguous()
    assert fits(num_seqs, num_heads)

    load_extension().kda_fused_decode(
        x,
        x.stride(0),
        conv_weight,
        conv_state,
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        raw_gate,
        raw_gate.stride(1),
        raw_beta,
        raw_beta.stride(1),
        A_log,
        dt_bias,
        state,
        state.stride(0),
        state_indices,
        state_indices.stride(0),
        num_accepted_tokens,
        query_start_loc,
        output_gate,
        output_gate.stride(0),
        norm_weight,
        output,
        output.stride(1),
        num_seqs,
        num_heads,
        float(lower_bound),
        float(HEAD_DIM**-0.5),
        float(norm_eps),
    )
    return output
