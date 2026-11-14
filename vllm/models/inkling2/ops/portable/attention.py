# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 paged relative attention without tml_fa4 or the Gluon decode
kernels: TokenSpeed's portable Triton kernel (`rel_mha`) over the vLLM KV
cache, reading an unsheared copy of the relative logits that
`attention_prologue` writes sheared for tml_fa4."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .rel_mha import triton_rel_mha_extend_with_kvcache


def use_portable_attention() -> bool:
    """Whether attention runs the portable kernels: tml_fa4 has no paged KV
    on SM12x and the decode/verify kernels use SM10x tcgen05."""
    return not current_platform.is_device_capability_family(100)


@triton.jit
def _unshear_rel_logits_kernel(
    Sheared,
    Out,
    CuSeqlensQ,
    SeqLens,
    stride_st,
    stride_sh,
    stride_ot,
    stride_oh,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """`Out[t, h, d]` = the sheared logit of distance `d` of token `t`.

    `qr_clip_rel_tokens` stores distance `diag - c` in column `c`, with
    `diag = pos + E + 256 - cdiv(pos - q_idx + tile_end, 128) * 128` for the
    token's 128-row query tile (clamped to `[E + 1, E + 255]`)."""
    seq = tl.program_id(0)
    q_idx = tl.program_id(1)
    head = tl.program_id(2)
    q_start = tl.load(CuSeqlensQ + seq)
    q_len = tl.load(CuSeqlensQ + seq + 1) - q_start
    if q_idx < q_len:
        t = (q_start + q_idx).to(tl.int64)
        pos = tl.load(SeqLens + seq) - q_len + q_idx
        tile_end = tl.minimum((q_idx // 128 + 1) * 128, q_len)
        E_PAD: tl.constexpr = E + 256
        diag = pos + E_PAD - tl.cdiv(pos - q_idx + tile_end, 128) * 128
        diag = tl.minimum(tl.maximum(diag, E + 1), E_PAD - 1)
        for start in tl.static_range(0, E, BLOCK_E):
            d = start + tl.arange(0, BLOCK_E)
            mask = d < E
            vals = tl.load(
                Sheared + t * stride_st + head * stride_sh + (diag - d), mask=mask
            )
            tl.store(Out + t * stride_ot + head * stride_oh + d, vals, mask=mask)


def unshear_rel_logits(
    sheared: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_reqs: int,
    max_query_len: int,
    num_tokens: int,
    extent: int,
) -> torch.Tensor:
    """`(num_tokens, H, extent)` logits by distance, from the `(T + 128, H,
    extent + 256)` tml_fa4 layout; rows past each request stay unwritten."""
    num_heads = sheared.shape[1]
    assert sheared.shape[2] == extent + 256 and sheared.stride(2) == 1
    out = sheared.new_empty((num_tokens, num_heads, extent))
    if num_reqs and max_query_len:
        _unshear_rel_logits_kernel[(num_reqs, max_query_len, num_heads)](
            sheared,
            out,
            query_start_loc,
            seq_lens,
            sheared.stride(0),
            sheared.stride(1),
            out.stride(0),
            out.stride(1),
            E=extent,
            BLOCK_E=min(1024, triton.next_power_of_2(extent)),
        )
    return out


def portable_rel_attention(
    q: torch.Tensor,
    sheared_rel_logits: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_reqs: int,
    max_query_len: int,
    max_seq_len: int,
    num_actual_tokens: int,
    extent: int,
    softmax_scale: float,
    window_left: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 `(T, H, D)` attention output and its `(T, H)` natural-log LSE
    for every request of the batch (prefill, decode and verify alike), as
    `Inkling2Attention._fa4_attention` produces them for `gate_sink`. Rows
    from `num_actual_tokens` (CUDA-graph padding) are zero."""
    num_tokens, num_heads, head_dim = q.shape
    out = q.new_zeros((num_tokens, num_heads, head_dim), dtype=torch.float32)
    lse = q.new_zeros((num_tokens, num_heads), dtype=torch.float32)
    nt = num_actual_tokens
    if nt == 0:
        return out, lse
    rel = unshear_rel_logits(
        sheared_rel_logits,
        query_start_loc,
        seq_lens,
        num_reqs,
        max_query_len,
        nt,
        extent,
    )
    # The cache views are `(blocks, block_size, kv_heads, D)`; TokenSpeed
    # flattens the first two, which needs block-contiguous token rows.
    assert key_cache.stride(0) == key_cache.shape[1] * key_cache.stride(1), (
        "portable Inkling2 attention needs one KV head per rank (TP = KV heads)"
    )
    o, l = triton_rel_mha_extend_with_kvcache(
        q[:nt],
        query_start_loc,
        query_start_loc,
        key_cache,
        value_cache,
        block_table[:num_reqs],
        seq_lens[:num_reqs],
        max_query_len,
        max_seq_len,
        rel,
        window_left=window_left,
        return_lse=True,
        softmax_scale=softmax_scale,
        output_dtype=torch.float32,
    )
    out[:nt].copy_(o)
    lse[:nt].copy_(l)
    return out, lse
