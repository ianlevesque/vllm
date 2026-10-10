# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU lifecycle regressions; native transport and model gates remain required."""

import ast
import importlib.util
import os
import sys
import weakref
from concurrent.futures import Future
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]


@pytest.fixture
def context(monkeypatch):
    ps = ModuleType("vllm.distributed.parallel_state")

    class Group:
        def __init__(self, name, ranks, transport):
            self.unique_name = name
            self.ranks = ranks
            self.device_communicator = transport

    ps.GroupCoordinator = Group
    ps._groups = {}
    ps._group_name_counter = {"tp": 1}
    ps._INNER_DP_WORLD = object()
    ps.events = []
    ps.destroy_model_parallel = lambda: ps.events.append("model-groups-closed")
    ps.destroy_distributed_environment = lambda: ps.events.append("world-closed")
    distributed = ModuleType("vllm.distributed")
    distributed.parallel_state = ps
    vllm = ModuleType("vllm")
    vllm.distributed = distributed
    allocator = ModuleType("vllm.distributed.device_communicators.pynccl_allocator")
    allocator.is_symmetric_memory_enabled = lambda: False
    for name, module in [
        ("vllm", vllm),
        ("vllm.distributed", distributed),
        ("vllm.distributed.parallel_state", ps),
        (allocator.__name__, allocator),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("NCCL_NET", "IB")
    monkeypatch.setenv("NCCL_IB_RELEASE_ON_FINALIZE", "1")
    path = ROOT / "vllm/distributed/network_sleep.py"
    spec = importlib.util.spec_from_file_location("network_sleep_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, ps, allocator


def put(ps, name="tp:0", ranks=None, transport=None):
    group = ps.GroupCoordinator(name, [0, 1] if ranks is None else ranks, transport)
    ps._groups[name] = weakref.ref(group)
    return group


def test_restore_preserves_cached_engram_and_global_group_identity(context):
    module, ps, _ = context
    old = put(ps, transport=SimpleNamespace(generation="old"))
    cached_engram = SimpleNamespace(group=old)
    fresh = put(ps, transport=SimpleNamespace(generation="fresh"))
    ps._TP = fresh
    module.restore_group_references({"tp:0": old})
    assert ps._TP is old and ps._groups["tp:0"]() is old
    assert cached_engram.group.device_communicator.generation == "fresh"


@pytest.mark.parametrize("change", ["name", "ranks"])
def test_restore_rejects_topology_change_without_rebinding(context, change):
    module, ps, _ = context
    old = put(ps)
    ps._groups.clear()
    fresh = put(
        ps,
        name="ep:0" if change == "name" else "tp:0",
        ranks=[0, 2] if change == "ranks" else [0, 1],
    )
    ps._TP = fresh
    with pytest.raises(RuntimeError):
        module.restore_group_references({"tp:0": old})
    assert ps._TP is fresh


def test_close_transport_waits_for_real_nccl_destroy_before_group_teardown(context):
    module, ps, _ = context
    comm = SimpleNamespace(available=True, disabled=False, comm=42)
    comm.nccl = SimpleNamespace(ncclCommDestroy=lambda ptr: ps.events.append(ptr))
    group = put(ps, transport=SimpleNamespace(pynccl_comm=comm))
    module.close_transport({"tp:0": group})
    assert ps.events == [42, "model-groups-closed", "world-closed"]
    assert not comm.available and comm.disabled
    assert not ps._groups and not ps._group_name_counter
    assert ps._INNER_DP_WORLD is None


def test_failed_nccl_destroy_does_not_advertise_closed_transport(context):
    module, ps, _ = context

    def fail(_):
        raise RuntimeError("NCCL cleanup failed")

    comm = SimpleNamespace(
        available=True,
        disabled=False,
        comm=42,
        nccl=SimpleNamespace(ncclCommDestroy=fail),
    )
    group = put(ps, transport=SimpleNamespace(pynccl_comm=comm))
    with pytest.raises(RuntimeError):
        module.close_transport({"tp:0": group})
    assert comm.available and not comm.disabled and not ps.events
    assert ps._groups["tp:0"]() is group


def test_native_roce_closes_before_nccl_and_is_not_closed_twice(context):
    module, ps, _ = context
    comm = SimpleNamespace(available=True, disabled=False, comm=42)
    comm.nccl = SimpleNamespace(ncclCommDestroy=lambda ptr: ps.events.append(ptr))
    native = SimpleNamespace(close=lambda: ps.events.append("roce-closed"))
    transport = SimpleNamespace(pynccl_comm=comm, b12x_ar_comm=native)
    group = put(ps, transport=transport)
    assert module.check_network_sleep(SimpleNamespace(use_v2_model_runner=True, rank=0))
    module.close_transport({"tp:0": group})
    assert ps.events == ["roce-closed", 42, "model-groups-closed", "world-closed"]
    assert transport.b12x_ar_comm is None


def test_failed_native_close_keeps_nccl_and_group_ownership(context):
    module, ps, _ = context

    def fail():
        raise RuntimeError("Native RoCE closure failed")

    native = SimpleNamespace(close=fail)
    comm = SimpleNamespace(available=True, disabled=False, comm=42)
    comm.nccl = SimpleNamespace(ncclCommDestroy=lambda ptr: ps.events.append(ptr))
    transport = SimpleNamespace(pynccl_comm=comm, b12x_ar_comm=native)
    group = put(ps, transport=transport)
    with pytest.raises(RuntimeError, match="Native RoCE closure failed"):
        module.close_transport({"tp:0": group})
    assert transport.b12x_ar_comm is native
    assert comm.available and not comm.disabled and not ps.events
    assert ps._groups["tp:0"]() is group


@pytest.mark.parametrize(
    "unsupported",
    ["empty", "symm", "custom", "v1", "socket", "retained-contexts", "stateless"],
)
def test_preflight_rejects_unsupported_transport_before_mutation(
    context, monkeypatch, unsupported
):
    module, ps, allocator = context
    group = put(ps)
    worker = SimpleNamespace(use_v2_model_runner=True, rank=0)
    if unsupported == "empty":
        ps._groups.clear()
    elif unsupported == "symm":
        allocator.is_symmetric_memory_enabled = lambda: True
    elif unsupported == "custom":
        group.device_communicator = SimpleNamespace(ca_comm=object())
    elif unsupported == "v1":
        worker.use_v2_model_runner = False
    elif unsupported == "socket":
        monkeypatch.setenv("NCCL_NET", "Socket")
    elif unsupported == "retained-contexts":
        monkeypatch.delenv("NCCL_IB_RELEASE_ON_FINALIZE")
    else:

        class Stateless(ps.GroupCoordinator):
            pass

        group = Stateless("tp:0", [0, 1], None)
        ps._groups["tp:0"] = weakref.ref(group)
    with pytest.raises(RuntimeError):
        module.check_network_sleep(worker)
    assert not ps.events


def core_method(name):
    path = ROOT / "vllm/v1/engine/core.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "EngineCore"
    )
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    scope = {
        "Future": Future,
        "Any": object,
        "PauseMode": str,
        "envs": SimpleNamespace(VLLM_SLEEP_RELEASE_TRANSPORT=True),
        "logger": SimpleNamespace(info=lambda *args: None),
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def test_network_sleep_waits_for_scheduler_drain_and_keeps_cache():
    events = []
    pause = Future()
    executor = SimpleNamespace(collective_rpc=lambda method: events.append(method))

    def pause_scheduler(**kwargs):
        assert kwargs == {"mode": "wait", "clear_cache": False}
        return pause

    engine = SimpleNamespace(model_executor=executor, pause_scheduler=pause_scheduler)
    result = core_method("sleep")(engine, level=0, mode="wait")
    assert events == ["check_network_sleep"] and not result.done()
    pause.set_result(None)
    assert result.done() and events == ["check_network_sleep", "sleep_network"]


def test_failed_drain_never_closes_transport():
    events = []
    pause = Future()
    engine = SimpleNamespace(
        model_executor=SimpleNamespace(collective_rpc=lambda m: events.append(m)),
        pause_scheduler=lambda **kwargs: pause,
    )
    result = core_method("sleep")(engine, level=0, mode="wait")
    pause.set_exception(RuntimeError("drain failed"))
    with pytest.raises(RuntimeError, match="drain failed"):
        result.result()
    assert events == ["check_network_sleep"]


def test_wake_rebuilds_transport_before_scheduler_accepts_work():
    events = []
    engine = SimpleNamespace(
        model_executor=SimpleNamespace(collective_rpc=lambda m: events.append(m)),
        resume_scheduler=lambda: events.append("admission"),
    )
    assert core_method("wake_up")(engine, tags=["scheduling"])
    assert events == ["wake_network", "admission"]


def test_failed_transport_restore_does_not_resume_scheduler():
    events = []

    def fail(_):
        raise RuntimeError("restore failed")

    engine = SimpleNamespace(
        model_executor=SimpleNamespace(collective_rpc=fail),
        resume_scheduler=lambda: events.append("admission"),
    )
    with pytest.raises(RuntimeError, match="restore failed"):
        core_method("wake_up")(engine)
    assert not events


def worker_methods(scope):
    path = ROOT / "vllm/v1/worker/gpu_worker.py"
    cls = next(
        n for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "Worker"
    )
    names = {"network_sleep_status", "sleep_network", "wake_network"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    extracted = ast.ClassDef(
        name="Worker", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    tree = ast.fix_missing_locations(ast.Module(body=[extracted], type_ignores=[]))
    scope.update(Any=object, os=os, logger=SimpleNamespace(info=lambda *args: None))
    exec(compile(tree, str(path), "exec"), scope)
    return scope["Worker"]()


def test_real_worker_sleep_wake_synchronizes_and_retains_model_and_cache(
    context, monkeypatch
):
    module, _, _ = context
    events = []
    groups = {"tp:0": object()}
    module.check_network_sleep = lambda worker: events.append("check")
    module.live_groups = lambda: groups
    module.clear_collective_graphs = lambda worker: events.append("graphs-released")
    module.close_transport = lambda original: events.append(("close", original))
    module.restore_group_references = lambda original: events.append(("restore", original))
    monkeypatch.setitem(sys.modules, "vllm.distributed.network_sleep", module)
    worker = worker_methods({
        "torch": SimpleNamespace(accelerator=SimpleNamespace(
            synchronize=lambda: events.append("synchronize")
        )),
        "set_current_vllm_config": lambda config: nullcontext(),
        "init_worker_distributed_environment": lambda *args, **kwargs:
            events.append(("init", kwargs["network_wake_generation"])),
        "current_platform": SimpleNamespace(dist_backend="nccl"),
    })
    worker.rank, worker.local_rank, worker.distributed_init_method = 3, 0, "tcp://host:1"
    worker.vllm_config = object()
    worker.model_runner = SimpleNamespace(
        model=object(), kv_cache=object(), capture_model=lambda: events.append("capture")
    )
    model, cache = worker.model_runner.model, worker.model_runner.kv_cache
    # Worker has no synchronize_device method; execute its actual lifecycle bodies.
    for generation in (1, 2):
        events.clear()
        slept = worker.sleep_network()
        assert slept == {"state": "sleeping", "rank": 3, "pid": os.getpid(),
                         "generation": generation}
        assert worker.sleep_network() == slept
        assert events == ["check", "synchronize", "graphs-released", ("close", groups)]
        events.clear()
        woke = worker.wake_network()
        assert woke["state"] == "active" and woke["generation"] == generation
        assert worker.wake_network() == woke
        assert events == [("init", generation), ("restore", groups), "capture", "synchronize"]
        assert worker.model_runner.model is model and worker.model_runner.kv_cache is cache


def test_wake_store_isolates_old_keys_and_separate_sleep_generations(
    context, monkeypatch
):
    module, _, _ = context
    shared = {}

    class Store:
        def set_timeout(self, timeout):
            self.timeout = timeout

    class PrefixStore:
        def __init__(self, prefix, store):
            self.prefix, self.store = prefix, store

        def set(self, key, value):
            shared[self.prefix + "/" + key] = value

        def get(self, key):
            return shared[self.prefix + "/" + key]

    dist = ModuleType("torch.distributed")
    dist.PrefixStore = PrefixStore
    torch = ModuleType("torch")
    torch.distributed = dist
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    store = Store()
    shared["endpoint"] = "dead listener"
    first = module.transport_rendezvous_store(
        "tcp://example:25000", 0, 2, timedelta(seconds=3), 1, store
    )
    second = module.transport_rendezvous_store(
        "tcp://example:25000", 0, 2, timedelta(seconds=3), 2, store
    )
    with pytest.raises(KeyError):
        first.get("endpoint")
    first.set("endpoint", "first wake listener")
    with pytest.raises(KeyError):
        second.get("endpoint")
    second.set("endpoint", "second wake listener")
    assert first.get("endpoint") == "first wake listener"
    assert second.get("endpoint") == "second wake listener"
    assert shared["endpoint"] == "dead listener"
    with pytest.raises(ValueError, match="positive generation"):
        module.transport_rendezvous_store(
            "tcp://example:25000", 0, 2, timedelta(seconds=3), 0, store
        )
