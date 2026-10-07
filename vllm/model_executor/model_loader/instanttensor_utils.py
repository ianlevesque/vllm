# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional bulk-loading communicator, independent of inference groups."""

import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


@contextmanager
def instanttensor_loading_group(world_group: Any) -> Generator[Any, None, None]:
    """Own a temporary NCCL group only when explicitly configured.

    VLLM_INSTANTTENSOR_NCCL_CTAS sets that group's min/max CTAs. The default
    continues to use the existing world device group. Process-wide transport,
    channel and buffer settings are never changed, and inference groups are
    neither mutated nor destroyed. The selected Torch/NCCL build must support
    per-communicator configuration; there is no fallback to global tuning.
    """
    setting = os.getenv("VLLM_INSTANTTENSOR_NCCL_CTAS")
    ctas = None
    if setting is not None:
        try:
            ctas = int(setting)
        except ValueError as exc:
            raise ValueError(
                "VLLM_INSTANTTENSOR_NCCL_CTAS must be an integer in 1..32"
            ) from exc
        if not 1 <= ctas <= 32:
            raise ValueError("VLLM_INSTANTTENSOR_NCCL_CTAS must be an integer in 1..32")

    if world_group is None or world_group.world_size <= 1:
        yield None
        return
    if ctas is None:
        yield world_group.device_group
        return

    import torch.distributed as dist

    # A device-bound default group can make new_group split its communicator.
    # Such a child's channels are capped by its parent's channel count.
    if dist.group.WORLD.bound_device_id is not None:
        raise ValueError(
            "isolated InstantTensor loading requires an unbound default group; "
            "set VLLM_DISTRIBUTED_USE_SPLIT_GROUP=0"
        )
    options = dist.ProcessGroupNCCL.Options()
    options.config.min_ctas = ctas
    options.config.max_ctas = ctas
    group = dist.new_group(
        list(world_group.ranks),
        backend="nccl",
        pg_options=options,
        timeout=world_group.device_group.options._timeout,
        group_desc="instanttensor:loading",
    )
    try:
        yield group
    finally:
        dist.destroy_process_group(group)
