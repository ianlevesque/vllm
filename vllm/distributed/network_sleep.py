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
from datetime import timedelta
from typing import Any


def transport_rendezvous_store(
    init_method: str,
    rank: int,
    world_size: int,
    timeout: timedelta,
    generation: int,
    store: Any = None,
) -> Any:
    """Isolate each wake from stale keys in a still-live multi-tenant store.

    Follow upstream stateless_init_torch_distributed_process_group's
    PrefixStore isolation. The executor can retain the original TCPStore;
    destroying a process group does not delete its Gloo rendezvous keys.
    """
    import torch.distributed as dist

    if generation < 1:
        raise ValueError("Network wake requires a positive generation")
    if store is None:
        store, _, _ = next(
            dist.rendezvous(init_method, rank, world_size, timeout=timeout)
        )
    store.set_timeout(timeout)
    return dist.PrefixStore(f"vllm-network-wake-{generation}", store)


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
            raise RuntimeError("Network sleep requires releasable NCCL/RoCEnante collectives")
    return {"groups": sorted(groups), "rank": worker.rank}


def clear_collective_graphs(worker: Any) -> None:
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

    # Reuse this K3 lineage's V2 profiling primitive. Upstream elastic EP
    # calls it release_graphs(), but this runner uses clear(), including
    # the ModelCudaGraphManager output buffers. Capture descriptions remain.
    gc.unfreeze()
    CUDAGraphWrapper.clear_all_graphs()
    BreakableCUDAGraphWrapper.clear_all_graphs()
    runner = worker.model_runner
    seen = set()
    for owner in (runner, getattr(runner, "speculator", None), runner.model_state):
        for manager in vars(owner).values() if owner is not None else ():
            if isinstance(manager, CudaGraphManager) and id(manager) not in seen:
                manager.clear()
                seen.add(id(manager))
    # Its V2 EncoderRunner has no graph manager or clear() method. Preserve
    # encoder embeddings/cache; compiled wrappers were released above.
    gc.collect()


def close_transport(groups: dict[str, Any]) -> None:
    from vllm.distributed import parallel_state as ps

    # K3's native RoCE adapter owns flat and hierarchical verbs resources,
    # plus raw Gloo exchange groups. Close those collectively while their
    # exchange groups still exist, before destroying NCCL or coordinators.
    # Clear only after successful close so a partial failure stays visible.
    for group in groups.values():
        dc = group.device_communicator
        native = getattr(dc, "b12x_ar_comm", None)
        if native is not None:
            native.close()
            dc.b12x_ar_comm = None

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
