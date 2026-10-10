# SPDX-License-Identifier: Apache-2.0
"""Fused sigmoid + biased top-16 routing for Kimi-K3 (bit-exact, JIT-built)."""

from functools import lru_cache
from pathlib import Path

import torch

_EXPERTS = 896
_TOPK = 16


@lru_cache(maxsize=1)
def _load_extension():
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("topk16.cu")
    return load(
        name="vllm_kimi_topk16_ext",
        sources=[str(source)],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def kimi_topk16_sigmoid(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    is_padding: torch.Tensor | None = None,
    *,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(topk_weights, topk_ids)`` for FP32 [rows, 896] logits.

    Bit-exact with vLLM's moeSigmoid + moeTopK pair (same __expf sigmoid,
    same lower-index tie break, same selection-order renormalization).
    """
    rows = router_logits.shape[0]
    weights = torch.empty(
        (rows, _TOPK), dtype=torch.float32, device=router_logits.device
    )
    indices = torch.empty((rows, _TOPK), dtype=torch.int32, device=router_logits.device)
    _load_extension().topk16_sigmoid(
        router_logits,
        correction_bias,
        weights,
        indices,
        is_padding,
        renormalize,
        routed_scaling_factor,
    )
    return weights, indices
