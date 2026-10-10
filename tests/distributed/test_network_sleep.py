# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU lifecycle regressions; native transport and model gates remain required."""

import ast
import importlib.util
import sys
import weakref
from concurrent.futures import Future
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("timeout", [None, timedelta(seconds=123)])
@pytest.mark.parametrize("backend,available", [("nccl", True), ("gloo", True), ("nccl", False)])
def test_actual_distributed_wake_resolves_backend_timeout(monkeypatch, timeout, backend, available):
    """Execute the installed initialization body with production's unset timeout."""
    path = ROOT / "vllm/distributed/parallel_state.py"
    fn = next(n for n in ast.parse(path.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == "init_distributed_environment")
    config = ModuleType("vllm.config")
    config.get_current_vllm_config_or_none = lambda: None
    network = ModuleType("vllm.distributed.network_sleep")
    calls, defaults = [], []
    store = object()
    expected_backend = backend if available else "gloo"
    default_timeout = timedelta(minutes=10 if expected_backend == "nccl" else 30)

    def resolve_default(selected):
        defaults.append(selected)
        return default_timeout

    def rendezvous(*args):
        calls.append(args)
        return store

    network.transport_rendezvous_store = rendezvous
    monkeypatch.setitem(sys.modules, "vllm.config", config)
    monkeypatch.setitem(sys.modules, "vllm.distributed.network_sleep", network)
    initialized = []
    scope = {
        "timedelta": timedelta,
        "logger": SimpleNamespace(**{n: lambda *a: None for n in ("debug", "info", "warning")}),
        "Backend": str, "_get_default_timeout": resolve_default,
        "torch": SimpleNamespace(distributed=SimpleNamespace(
            is_initialized=lambda: False, is_backend_available=lambda b: available,
            is_gloo_available=lambda: True, get_world_size=lambda: 16,
            init_process_group=lambda **kwargs: initialized.append(kwargs))),
        "envs": SimpleNamespace(VLLM_DISTRIBUTED_USE_SPLIT_GROUP=False),
        "_WORLD": None, "_NODE_COUNT": None, "_INNER_DP_WORLD": None,
        "init_world_group": lambda *a: SimpleNamespace(cpu_group=object()),
        "_node_count": lambda group: 16,
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), scope)
    scope[fn.name](16, 3, "tcp://example:25000", 0, backend, timeout, 2)
    assert calls == [("tcp://example:25000", 3, 16,
                      timeout if timeout is not None else default_timeout, 2, None)]
    assert defaults == ([expected_backend] if timeout is None else [])
    assert initialized == [dict(backend=expected_backend, init_method=None, store=store,
                               world_size=16, rank=3, timeout=timeout)]


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
