# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU lifecycle regressions; native transport and model gates remain required."""

import ast
import importlib.util
import sys
import weakref
from concurrent.futures import Future
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
