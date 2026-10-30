# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Filter out non-local expert weights during loading to avoid redundant I/O.

In DP+EP deployments each rank only needs its own expert shard.  Skipping
non-local expert tensors *before* they are read from disk eliminates the
majority of storage I/O for MoE models (experts typically account for
~85-90 % of total weight bytes).
"""

import regex as re

# Matches per-expert weight names like ".experts.42.gate_proj.weight".
# Does NOT match 3D fused-expert names like ".experts.gate_proj.weight"
# (no numeric id) — those are intentionally left unfiltered so the full
# tensor is loaded and sliced later by RoutedExperts.weight_loader.
_EXPERT_ID_RE = re.compile(r"\.experts\.(\d+)\.")
_WEIGHT_TENSOR_SUFFIXES = (".weight", ".weight_packed")
_BLOCK_WEIGHT_SCALE_SUFFIXES = (
    ".weight_scale",
    ".weight_scale_inv",
    ".qscale_weight",
    "_weight_scale",
    "_weight_scale_inv",
    "_qscale_weight",
)


def parse_expert_id(weight_name: str) -> int | None:
    """Return the expert id embedded in *weight_name*, or ``None`` if it is
    not an per-expert weight.

    Returns ``None`` for dense weights (attention, layernorm, embedding),
    shared experts, and 3D fused-expert tensors where all experts are stored
    in a single tensor without a numeric expert id in the name."""
    m = _EXPERT_ID_RE.search(weight_name)
    return int(m.group(1)) if m else None


def compute_local_expert_ids(
    num_experts: int,
    ep_size: int,
    ep_rank: int,
    placement: str = "linear",
) -> set[int] | None:
    """Compute the set of global expert ids owned by *ep_rank*.

    Returns ``None`` when EP is not active (``ep_size <= 1``), meaning all
    experts are local and no filtering should be performed.

    The distribution logic mirrors
    :func:`vllm.model_executor.layers.fused_moe.layer.determine_expert_map`.

    Args:
        num_experts: Total number of experts in the layer.
        ep_size: Expert parallel world size.
        ep_rank: Rank whose local expert ids are computed.
        placement: ``"linear"`` for contiguous assignment,
            ``"round_robin"`` for interleaved assignment.

    """
    if ep_size <= 1:
        return None

    if placement == "linear":
        base = num_experts // ep_size
        remainder = num_experts % ep_size
        start = ep_rank * base + min(ep_rank, remainder)
        local_count = base + (1 if ep_rank < remainder else 0)
        return set(range(start, start + local_count))
    elif placement == "round_robin":
        return set(range(ep_rank, num_experts, ep_size))
    else:
        raise ValueError(f"Unknown expert placement strategy: {placement}")


def should_skip_weight(
    weight_name: str,
    local_expert_ids: set[int] | None,
) -> bool:
    """Return ``True`` if *weight_name* is an expert weight that does not
    belong to the local rank and should be skipped during loading."""
    if local_expert_ids is None:
        return False
    eid = parse_expert_id(weight_name)
    if eid is None:
        # Not an expert weight (dense / shared-expert / embedding) → keep.
        return False
    # Skip the heavy per-expert tensors RoutedExperts ignores for a
    # non-local expert: weights and block weight scales. Keep scalar
    # activation/global scales (``input_scale``, ``weight_scale_2`` and
    # ``*_global_scale``), which some NVFP4 backends aggregate across all
    # experts before discarding the non-local weights.
    if not weight_name.endswith(
        _WEIGHT_TENSOR_SUFFIXES + _BLOCK_WEIGHT_SCALE_SUFFIXES
    ):
        return False
    return eid not in local_expert_ids
