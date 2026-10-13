# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense appended-logit oracle; independent of the online-softmax kernel."""

import torch

SEED = 20261013
HEAD_DIM = 64
# name, global Q heads, global KV heads, TP size, rank, window, packed lengths.
# TP8 entries exercise local shapes/slicing, not distributed communication.
CASES = [
    ("small-mha", 4, 4, 1, 0, 8, (5, 37)),
    ("small-gqa", 4, 2, 1, 0, 8, (5, 37)),
    ("production-tp1", 32, 8, 1, 0, 64, (5, 137)),
    ("production-tp8-rank0", 32, 8, 8, 0, 64, (5, 137)),
    ("production-tp8-rank3", 32, 8, 8, 3, 64, (5, 137)),
    ("production-tp8-rank7", 32, 8, 8, 7, 64, (5, 137)),
]


def make_inputs(case, dtype, device="cpu"):
    _, heads, kv_heads, tp, rank, window, lengths = case
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    total = sum(lengths)

    def draw(nheads, offset):
        tensor = torch.randn(total, nheads, HEAD_DIM, generator=generator)
        return (tensor * 0.3 + offset).to(device=device, dtype=dtype)

    q, k, v = draw(heads // tp, 0), draw(kv_heads // tp, 0), draw(kv_heads // tp, 1)
    sinks = torch.linspace(2, 4, heads).to(device=device, dtype=dtype)
    cu = torch.tensor([0, lengths[0], total], device=device, dtype=torch.int32)
    start = rank * (heads // tp)
    return q, k, v, cu, sinks, sinks[start : start + heads // tp], window


def dense_reference(q, k, v, cu, sinks, scale, window, mode="denominator"):
    assert mode in {"denominator", "key0", "none"}
    groups = q.shape[1] // k.shape[1]
    out = torch.empty_like(q, dtype=torch.float32)
    boundaries = cu.tolist()
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        qs = q[start:end].float()
        ks = k[start:end].float().repeat_interleave(groups, dim=1)
        vs = v[start:end].float().repeat_interleave(groups, dim=1)
        scores = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        pos = torch.arange(end - start, device=q.device)
        scores.masked_fill_((pos[:, None] - pos[None, :]).abs() > window, -torch.inf)
        if mode == "denominator":
            null_logits = sinks.float()[:, None, None].expand(-1, end - start, 1)
            scores = torch.cat([scores, null_logits], dim=-1)
            vs = torch.cat([vs, torch.zeros_like(vs[:1])], dim=0)
        elif mode == "key0":
            scores[..., 0] += sinks.float()[:, None]
        out[start:end] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vs)
    return out


def relative_error(actual, expected):
    return ((actual.float() - expected).norm() / expected.norm()).item()


def assert_discriminating(q, k, v, cu, sinks, scale, window, reference):
    errors = {}
    for mode in ("key0", "none"):
        wrong = dense_reference(q, k, v, cu, sinks, scale, window, mode)
        errors[mode] = relative_error(wrong, reference)
        assert errors[mode] > 0.05, (mode, errors[mode])
        # The last query cannot see key zero: its masked bias cannot be a sink.
        tail_error = relative_error(wrong[-1:], reference[-1:])
        assert tail_error > 0.05, (mode, "tail", tail_error)
    return errors
