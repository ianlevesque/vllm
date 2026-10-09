# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport teardown for drained level-0 sleep, adapted from upstream #35934.

Unlike ncclCommSuspend(NCCL_SUSPEND_MEM), this releases RDMA transports.
Model allocations stay resident; graphs must be rebuilt before scheduling resumes.
"""

from __future__ import annotations

import gc
import os
import weakref
from typing import Any


def live_groups() -> dict[str, Any]:
    from vllm.distributed import parallel_state as ps

    return {name: group for name, ref in ps._groups.items() if (group := ref())}


def check_network_sleep(worker: Any) -> dict[str, Any]:
    from vllm.distributed import parallel_state as ps
    from vllm.distributed.device_communicators.pynccl_allocator import (
        is_symmetric_memory_enabled,
    )

    if not worker.use_v2_model_runner:
        raise RuntimeError("Network sleep requires the V2 model runner")
    if os.environ.get("NCCL_NET") != "IB":
        raise RuntimeError("Network sleep requires the configured IB/RoCE transport")
    if os.environ.get("NCCL_IB_RELEASE_ON_FINALIZE") != "1":
        raise RuntimeError("Network sleep requires opt-in NCCL IB context release")
    if is_symmetric_memory_enabled():
        raise RuntimeError("Network sleep cannot retain NCCL symmetric allocations")
    groups = live_groups()
    if not groups or any(type(g) is not ps.GroupCoordinator for g in groups.values()):
        raise RuntimeError("Network sleep requires static GroupCoordinators")
    for group in groups.values():
        dc = group.device_communicator
        if dc is not None and any(
            getattr(dc, name, None) is not None
            for name in (
                "ca_comm",
                "fi_ar_comm",
                "fi_pcie_ipc_ar_comm",
                "symm_mem_comm",
            )
        ):
            raise RuntimeError("Network sleep requires ordinary NCCL collectives")
    return {"groups": sorted(groups), "rank": worker.rank}


def clear_collective_graphs(worker: Any) -> None:
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

    # Reuse the V2 graph-release primitive used by elastic EP. Preserve the
    # capture descriptions and compiled model while dropping stale NCCL handles.
    gc.unfreeze()
    CUDAGraphWrapper.clear_all_graphs()
    BreakableCUDAGraphWrapper.clear_all_graphs()
    runner = worker.model_runner
    seen = set()
    for owner in (runner, getattr(runner, "speculator", None), runner.model_state):
        for manager in vars(owner).values() if owner is not None else ():
            if isinstance(manager, CudaGraphManager) and id(manager) not in seen:
                manager.release_graphs()
                seen.add(id(manager))
    encoder = getattr(runner.model_state, "encoder_runner", None)
    if encoder is not None:
        encoder.clear()
    gc.collect()


def close_transport(groups: dict[str, Any]) -> None:
    from vllm.distributed import parallel_state as ps

    # The generic shutdown path uses a timed daemon-thread ncclCommAbort.
    # NIC power-off needs actual synchronous completion, after graph release.
    for group in groups.values():
        dc = group.device_communicator
        comm = getattr(dc, "pynccl_comm", None)
        if comm is not None and comm.available and not comm.disabled:
            comm.nccl.ncclCommDestroy(comm.comm)
            comm.available = False
            comm.disabled = True
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()
    ps._INNER_DP_WORLD = None
    ps._group_name_counter.clear()
    ps._groups.clear()
    gc.collect()


def restore_group_references(previous: dict[str, Any]) -> None:
    from vllm.distributed import parallel_state as ps

    current = live_groups()
    if current.keys() != previous.keys():
        raise RuntimeError("Distributed group topology changed during network sleep")
    replacements = {}
    for name, fresh in current.items():
        original = previous[name]
        if fresh.ranks != original.ranks or fresh.unique_name != original.unique_name:
            raise RuntimeError("Distributed ranks changed during network sleep")
        replacements[id(fresh)] = original
    # Engram storage and model layers can retain GroupCoordinator references.
    # Preserve those objects while replacing all of their transport state.
    for name, fresh in current.items():
        original = previous[name]
        original.__dict__.clear()
        original.__dict__.update(fresh.__dict__)
        ps._groups[name] = weakref.ref(original)
    for name, value in list(vars(ps).items()):
        replacement = replacements.get(id(value))
        if replacement is not None:
            setattr(ps, name, replacement)
