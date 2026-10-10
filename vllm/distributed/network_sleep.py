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


def worker_global_rank(worker: Any) -> int:
    """Keep the across-DP identity available while all process groups are gone."""
    from vllm.distributed import parallel_state as ps

    if not hasattr(worker, "_network_sleep_rank"):
        worker._network_sleep_rank = ps.get_world_group().rank
    return worker._network_sleep_rank


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
    config = worker.vllm_config
    if config.kv_transfer_config is not None or config.ec_transfer_config is not None:
        raise RuntimeError(
            "Network sleep requires inactive KV/encoder transfer connectors"
        )
    if getattr(worker, "weight_transfer_engine", None) is not None:
        raise RuntimeError("Network sleep requires inactive weight-transfer transports")
    if config.parallel_config.use_ubatching:
        raise RuntimeError("Network sleep has not qualified dual-batch overlap")
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
                "qr_comm",
                "aiter_ar_comm",
            )
        ):
            raise RuntimeError("Network sleep requires ordinary NCCL collectives")
        manager = getattr(dc, "all2all_manager", None)
        if manager is not None:
            from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager

            if type(manager) is not AgRsAll2AllManager:
                raise RuntimeError(
                    "Network sleep requires releasable AgRs expert collectives"
                )
    return {"groups": sorted(groups), "rank": worker_global_rank(worker)}


def clear_collective_graphs(worker: Any) -> None:
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    from vllm.platforms import current_platform

    # Reuse the V2 graph-release primitive used by elastic EP. Preserve the
    # capture descriptions and compiled model while dropping stale NCCL handles.
    gc.unfreeze()
    CUDAGraphWrapper.clear_all_graphs()
    BreakableCUDAGraphWrapper.clear_all_graphs()
    runner = worker.model_runner
    seen = set()
    managers = []
    for owner in (runner, getattr(runner, "speculator", None), runner.model_state):
        for manager in vars(owner).values() if owner is not None else ():
            if isinstance(manager, CudaGraphManager) and id(manager) not in seen:
                manager.release_graphs()
                managers.append(manager)
                seen.add(id(manager))
    encoder = getattr(runner.model_state, "encoder_runner", None)
    if encoder is not None:
        encoder.clear()
    gc.collect()
    # A pool whose last graph was released cannot be revived while retained
    # tensors still reference it. Follow upstream disposable-profile isolation,
    # rebinding all surviving owners to the same fresh shared capture pool.
    pool = current_platform.graph_pool_handle()
    type(current_platform)._global_graph_pool = pool
    for manager in managers:
        if manager.pool is not None:
            manager.pool = pool
    for wrapper_cls in (CUDAGraphWrapper, BreakableCUDAGraphWrapper):
        for wrapper in list(wrapper_cls._all_instances):
            if wrapper.graph_pool is not None:
                wrapper.graph_pool = pool


def close_transport(groups: dict[str, Any]) -> None:
    from vllm.distributed import parallel_state as ps

    # Native adapters own verbs handles and raw Gloo exchange groups. Join
    # their closure before destroying the groups on which their barriers rely.
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
    if ps._INNER_DP_WORLD is not None and ps._INNER_DP_WORLD is not ps._WORLD:
        ps._INNER_DP_WORLD.destroy()
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
    # Fresh AgRs managers retain TP/DP coordinator references of their own.
    # Rebind those to the canonical originals as well, so future generations
    # cannot leave shadow coordinators holding retired transports.
    for group in previous.values():
        dc = group.device_communicator
        for owner in (dc, getattr(dc, "all2all_manager", None)):
            if owner is not None:
                for name, value in list(vars(owner).items()):
                    replacement = replacements.get(id(value))
                    if replacement is not None:
                        setattr(owner, name, replacement)
    for name, value in list(vars(ps).items()):
        replacement = replacements.get(id(value))
        if replacement is not None:
            setattr(ps, name, replacement)


def close_engine_control_group(engine: Any) -> None:
    """Close the DP EngineCore's Gloo owner after distributed pause consensus.

    This group lives outside Worker GroupCoordinators. Retain its store and
    original process, but do not retain sockets bound to a disappearing NIC.
    """
    group = getattr(engine, "dp_group", None)
    if group is None:
        return
    from vllm.distributed.utils import stateless_destroy_torch_distributed_process_group

    if (
        engine.pending_pause
        or engine.engines_running
        or not engine.ignore_start_dp_wave
    ):
        raise RuntimeError(
            "DP network sleep requires completed all-engine pause consensus"
        )
    stateless_destroy_torch_distributed_process_group(group)
    engine.dp_group = None
    engine._network_sleep_dp_generation = (
        getattr(engine, "_network_sleep_dp_generation", 0) + 1
    )


def restore_engine_control_group(engine: Any) -> None:
    """Rebuild the original DP control topology before scheduling resumes."""
    if not hasattr(engine, "_network_sleep_dp_generation"):
        return
    if engine.dp_group is not None:
        raise RuntimeError("DP network wake found an already-active control group")
    import torch.distributed as dist
    from torch.distributed.distributed_c10d import Backend, _get_default_timeout

    from vllm.distributed.utils import init_gloo_process_group

    timeout = engine.vllm_config.parallel_config.cpu_distributed_timeout
    if timeout is None:
        timeout = _get_default_timeout(Backend("gloo"))
    store = dist.PrefixStore(
        f"vllm-network-dp-wake-{engine._network_sleep_dp_generation}",
        engine.dp_store,
    )
    store.set_timeout(timeout)
    engine.dp_group = init_gloo_process_group(
        prefix_store=store,
        group_rank=engine.dp_rank,
        group_size=engine.dp_size,
        timeout=timeout,
    )
