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


@pytest.mark.parametrize("capture_enabled", [False, True])
def test_graph_release_uses_installed_k3_manager_api_and_retains_cache(context, monkeypatch, capture_enabled):
    module, _, _ = context
    tree = ast.parse((ROOT / "vllm/v1/worker/gpu/cudagraph_utils.py").read_text())
    classes = []
    for name in ("CudaGraphManager", "ModelCudaGraphManager"):
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "clear"]
        classes.append(cls)
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[])), "installed_graph_clear", "exec"), namespace)
    manager = namespace["ModelCudaGraphManager"]()
    manager.graphs = {"captured": object()}
    manager._graphs_captured = True
    manager.breakable_cg_runner = object()
    manager.hidden_states = object()
    manager.aux_hidden_states = [object()]
    manager.intermediate_tensors = object()
    descriptors, pool, weights, kv, embeddings, encoder_cache = (object() for _ in range(6))
    manager._capture_descs, manager.pool = descriptors, pool if capture_enabled else None
    # The selected V2 EncoderRunner has no clear() or separate graph manager.
    encoder = SimpleNamespace(inputs_embeds=embeddings, encoder_cache=encoder_cache)
    runner = SimpleNamespace(cudagraph_manager=manager, speculator=SimpleNamespace(manager=manager),
                             model=weights, kv_caches=kv, model_state=SimpleNamespace(encoder_runner=encoder))
    wrappers = ModuleType("vllm.compilation.cuda_graph")
    compiled_wrapper = SimpleNamespace(graph_pool=pool if capture_enabled else None)
    wrappers.CUDAGraphWrapper = SimpleNamespace(clear_all_graphs=lambda: None, _all_instances=[compiled_wrapper])
    breakable = ModuleType("vllm.compilation.breakable_cudagraph")
    piecewise_wrapper = SimpleNamespace(graph_pool=pool if capture_enabled else None)
    breakable.BreakableCUDAGraphWrapper = SimpleNamespace(clear_all_graphs=lambda: None, _all_instances=[piecewise_wrapper])
    platforms = ModuleType("vllm.platforms")
    new_pool = object()
    class Platform:
        _global_graph_pool = pool
        graph_pool_handle = staticmethod(lambda: new_pool)
    platforms.current_platform = Platform()
    managers = ModuleType("vllm.v1.worker.gpu.cudagraph_utils")
    managers.CudaGraphManager = namespace["CudaGraphManager"]
    for item in (wrappers, breakable, managers, platforms):
        monkeypatch.setitem(sys.modules, item.__name__, item)
    module.clear_collective_graphs(SimpleNamespace(model_runner=runner))
    assert not manager.graphs and not manager._graphs_captured
    assert manager.hidden_states is None and not manager.aux_hidden_states
    assert manager.intermediate_tensors is None and manager.breakable_cg_runner is None
    assert manager._capture_descs is descriptors
    assert manager.pool is (new_pool if capture_enabled else None)
    assert compiled_wrapper.graph_pool is (new_pool if capture_enabled else None)
    assert piecewise_wrapper.graph_pool is (new_pool if capture_enabled else None)
    assert type(platforms.current_platform)._global_graph_pool is new_pool
    assert runner.model is weights and runner.kv_caches is kv
    assert runner.model_state.encoder_runner is encoder
    assert encoder.inputs_embeds is embeddings and encoder.encoder_cache is encoder_cache


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
