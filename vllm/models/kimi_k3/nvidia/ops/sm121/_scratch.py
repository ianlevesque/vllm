# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device scratch of the SM12x fused kernels.

The AttnRes kernel splits a row over the blocks of a thread-block cluster,
which exchange partial sums through a few device floats.  Two launches sharing
them must not overlap, so each call site (a decoder layer's AttnRes) owns its
scratch and allocates it when the module is built: never inside a CUDA graph
capture, whose private pool must not own memory that outlives the graph.
"""

import torch

KDA_MAX_GROUPS = 32  # (sequence, local head) groups per launch
# Rows one AttnRes launch supports (560 bytes of scratch per row).  The
# dispatch in ops/attn_res.py uses the kernel for decode batches only, where it
# is the faster one.
ATTN_RES_MAX_ROWS = 1024
ATTN_RES_BLOCKS = 7
ATTN_RES_PARTIALS = 20


def _check_not_capturing(what: str) -> None:
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"{what} must be allocated before CUDA graph capture")


class AttnResScratch:
    """Cross-block partial sums of one AttnRes call site (one cluster per row)."""

    def __init__(self, device: torch.device) -> None:
        _check_not_capturing("Kimi SM12x AttnRes scratch")
        self.partial = torch.zeros(
            ATTN_RES_MAX_ROWS * ATTN_RES_BLOCKS * ATTN_RES_PARTIALS,
            dtype=torch.float32,
            device=device,
        )
