# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU lifecycle regressions; native transport and model gates remain required."""

import ast
import asyncio
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
@pytest.mark.parametrize(
    "backend,available", [("nccl", True), ("gloo", True), ("nccl", False)]
)
def test_actual_distributed_wake_resolves_backend_timeout(
    monkeypatch, timeout, backend, available
):
    """Execute the installed initialization body with production's unset timeout."""
    path = ROOT / "vllm/distributed/parallel_state.py"
    fn = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "init_distributed_environment"
    )
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
        "logger": SimpleNamespace(
            **{n: lambda *a: None for n in ("debug", "info", "warning")}
        ),
        "Backend": str,
        "_get_default_timeout": resolve_default,
        "torch": SimpleNamespace(
            distributed=SimpleNamespace(
                is_initialized=lambda: False,
                is_backend_available=lambda b: available,
                is_gloo_available=lambda: True,
                get_world_size=lambda: 16,
                init_process_group=lambda **kwargs: initialized.append(kwargs),
            )
        ),
        "envs": SimpleNamespace(VLLM_DISTRIBUTED_USE_SPLIT_GROUP=False),
        "_WORLD": None,
        "_NODE_COUNT": None,
        "_INNER_DP_WORLD": None,
        "init_world_group": lambda *a: SimpleNamespace(cpu_group=object()),
        "_node_count": lambda group: 16,
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), scope)
    scope[fn.name](16, 3, "tcp://example:25000", 0, backend, timeout, 2)
    assert calls == [
        (
            "tcp://example:25000",
            3,
            16,
            timeout if timeout is not None else default_timeout,
            2,
            None,
        )
    ]
    assert defaults == ([expected_backend] if timeout is None else [])
    assert initialized == [
        dict(
            backend=expected_backend,
            init_method=None,
            store=store,
            world_size=16,
            rank=3,
            timeout=timeout,
        )
    ]


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
    ps._INNER_DP_WORLD = None
    ps._WORLD = None
    ps.get_world_group = lambda: SimpleNamespace(rank=0)
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
    monkeypatch.setitem(sys.modules, "vllm.distributed.network_sleep", module)
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
    worker = SimpleNamespace(
        use_v2_model_runner=True,
        rank=0,
        vllm_config=SimpleNamespace(
            kv_transfer_config=None,
            ec_transfer_config=None,
            parallel_config=SimpleNamespace(use_ubatching=False),
        ),
    )
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


def test_network_sleep_waits_for_scheduler_drain_and_keeps_cache(context):
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


def test_failed_drain_never_closes_transport(context):
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


def test_wake_rebuilds_transport_before_scheduler_accepts_work(context):
    events = []
    engine = SimpleNamespace(
        model_executor=SimpleNamespace(collective_rpc=lambda m: events.append(m)),
        resume_scheduler=lambda: events.append("admission"),
        _network_sleep_state="sleeping",
    )
    assert core_method("wake_up")(engine, tags=["scheduling"])
    assert events == ["wake_network", "admission"]


def test_failed_transport_restore_does_not_resume_scheduler(context):
    events = []

    def fail(_):
        raise RuntimeError("restore failed")

    engine = SimpleNamespace(
        model_executor=SimpleNamespace(collective_rpc=fail),
        resume_scheduler=lambda: events.append("admission"),
        _network_sleep_state="sleeping",
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


def test_worker_global_rank_survives_teardown(context):
    module, ps, _ = context
    workers = [SimpleNamespace(rank=i % 4) for i in range(16)]
    for rank, worker in enumerate(workers):
        ps.get_world_group = lambda rank=rank: SimpleNamespace(rank=rank)
        assert module.worker_global_rank(worker) == rank
    ps.get_world_group = lambda: (_ for _ in ()).throw(
        AssertionError("world is closed")
    )
    assert [module.worker_global_rank(w) for w in workers] == list(range(16))


def test_dp_rpc_returns_all_original_workers():
    path = ROOT / "vllm/v1/engine/core_client.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "AsyncMPClient"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "collective_rpc_async"
    )
    fn.decorator_list = []
    scope = {}
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            fn,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), scope)
    expected = [{"rank": i, "pid": 1000 + i, "state": "sleeping"} for i in range(16)]
    calls = []

    async def all_engines(*args):
        calls.append(args)
        return [expected[i : i + 4] for i in range(0, 16, 4)]

    result = asyncio.run(
        scope[fn.name](
            SimpleNamespace(call_utility_all_async=all_engines),
            "network_sleep_status",
            120,
        )
    )
    assert result == expected
    assert calls == [("collective_rpc", "network_sleep_status", 120, (), None)]


def test_dp_rpc_peer_failure_is_not_partial_success():
    path = ROOT / "vllm/v1/engine/core_client.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "AsyncMPClient"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "collective_rpc_async"
    )
    scope = {}
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            fn,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), scope)

    async def fail(*args):
        raise RuntimeError("DP3 failed")

    with pytest.raises(RuntimeError, match="DP3 failed"):
        asyncio.run(
            scope[fn.name](
                SimpleNamespace(call_utility_all_async=fail), "network_sleep_status"
            )
        )


def test_native_roce_closes_before_nccl_and_exchange_groups(context):
    module, ps, _ = context
    native = SimpleNamespace(close=lambda: ps.events.append("native"))
    comm = SimpleNamespace(available=True, disabled=False, comm=42)
    comm.nccl = SimpleNamespace(ncclCommDestroy=lambda ptr: ps.events.append("nccl"))
    dc = SimpleNamespace(b12x_ar_comm=native, pynccl_comm=comm)
    group = put(ps, transport=dc)
    ps._INNER_DP_WORLD = SimpleNamespace(destroy=lambda: ps.events.append("inner-dp"))
    module.close_transport({"tp:0": group})
    assert ps.events == [
        "native",
        "nccl",
        "model-groups-closed",
        "inner-dp",
        "world-closed",
    ]
    assert dc.b12x_ar_comm is None


@pytest.mark.parametrize("timeout", [None, timedelta(seconds=73)])
def test_dp_control_group_rebuilds_after_consensus_with_fresh_generation(
    context, monkeypatch, timeout
):
    module, _, _ = context
    events = []
    utils = ModuleType("vllm.distributed.utils")
    utils.stateless_destroy_torch_distributed_process_group = lambda group: (
        events.append(("close", group))
    )
    utils.init_gloo_process_group = lambda **kw: (
        events.append(("open", kw))
        or SimpleNamespace(generation=kw["prefix_store"].prefix)
    )
    monkeypatch.setitem(sys.modules, utils.__name__, utils)

    class Store:
        def __init__(self, prefix, base):
            self.prefix, self.base = prefix, base

        def set_timeout(self, value):
            self.timeout = value

    dist = ModuleType("torch.distributed")
    dist.PrefixStore = Store
    torch = ModuleType("torch")
    torch.distributed = dist
    backend = ModuleType("torch.distributed.distributed_c10d")
    backend.Backend = str
    backend._get_default_timeout = lambda backend: timedelta(minutes=30)
    for m in (torch, dist, backend):
        monkeypatch.setitem(sys.modules, m.__name__, m)
    original, store = object(), object()
    engine = SimpleNamespace(
        dp_group=original,
        dp_store=store,
        dp_rank=3,
        dp_size=4,
        pending_pause=False,
        engines_running=False,
        ignore_start_dp_wave=True,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(cpu_distributed_timeout=timeout)
        ),
    )
    for generation in (1, 2):
        old = engine.dp_group
        module.close_engine_control_group(engine)
        assert engine.dp_group is None and events[-1] == ("close", old)
        module.restore_engine_control_group(engine)
        restored = events[-1][1]
        assert restored["prefix_store"].base is store
        assert restored["prefix_store"].prefix == f"vllm-network-dp-wake-{generation}"
        assert restored["group_rank"] == 3 and restored["group_size"] == 4
        assert restored["timeout"] == (timeout or timedelta(minutes=30))
        assert engine.dp_store is store


@pytest.mark.parametrize(
    "flag,value",
    [
        ("pending_pause", True),
        ("engines_running", True),
        ("ignore_start_dp_wave", False),
    ],
)
def test_dp_control_group_requires_completed_consensus(
    context, monkeypatch, flag, value
):
    module, _, _ = context
    utils = ModuleType("vllm.distributed.utils")
    events = []
    utils.stateless_destroy_torch_distributed_process_group = lambda group: (
        events.append(group)
    )
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    group = object()
    engine = SimpleNamespace(
        dp_group=group,
        pending_pause=False,
        engines_running=False,
        ignore_start_dp_wave=True,
    )
    setattr(engine, flag, value)
    with pytest.raises(RuntimeError, match="consensus"):
        module.close_engine_control_group(engine)
    assert engine.dp_group is group and not events


def test_single_engine_has_no_dp_control_owner(context):
    module, _, _ = context
    engine = SimpleNamespace()
    module.close_engine_control_group(engine)
    module.restore_engine_control_group(engine)
    assert vars(engine) == {}


def test_graph_release_uses_current_apis_and_renews_retained_pools(
    context, monkeypatch
):
    module, _, _ = context
    events = []

    class Wrapper:
        _all_instances = []

        @classmethod
        def clear_all_graphs(cls):
            events.append("wrapper-clear")

    class Breakable(Wrapper):
        _all_instances = []

    wrapper = Wrapper()
    wrapper.graph_pool = object()
    Wrapper._all_instances.append(wrapper)
    path = ROOT / "vllm/v1/worker/gpu/cudagraph_utils.py"
    original_cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "CudaGraphManager"
    )
    fn = next(
        n
        for n in original_cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "release_graphs"
    )
    namespace = {"BreakableCUDAGraphWrapper": Breakable}
    only = ast.ClassDef(
        name="CudaGraphManager", bases=[], keywords=[], body=[fn], decorator_list=[]
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[only], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    Manager = namespace["CudaGraphManager"]
    first_pool, weights, kv, embeddings, cache = (object() for _ in range(5))
    target, draft, inner = [Manager() for _ in range(3)]
    for manager in (target, draft, inner):
        manager.graphs, manager._graphs_captured = {1: object()}, True
        manager._capture_descs = [object()]
        manager.pool, manager.breakable_cg_runner = first_pool, object()
    clear_encoder = lambda: events.append("encoder-clear")
    encoder = SimpleNamespace(
        clear=clear_encoder, inputs_embeds=embeddings, encoder_cache=cache
    )
    runner = SimpleNamespace(
        cudagraph_manager=target,
        speculator=SimpleNamespace(manager=draft),
        model=weights,
        kv_caches=kv,
        model_state=SimpleNamespace(inner=inner, encoder_runner=encoder),
    )

    class Platform:
        _global_graph_pool = first_pool

        def graph_pool_handle(self):
            return object()

    modules = {
        "vllm.compilation.cuda_graph": {"CUDAGraphWrapper": Wrapper},
        "vllm.compilation.breakable_cudagraph": {
            "BreakableCUDAGraphWrapper": Breakable
        },
        "vllm.v1.worker.gpu.cudagraph_utils": {"CudaGraphManager": Manager},
        "vllm.platforms": {"current_platform": Platform()},
    }
    for name, attrs in modules.items():
        mocked = ModuleType(name)
        vars(mocked).update(attrs)
        monkeypatch.setitem(sys.modules, name, mocked)
    for cycle in range(2):
        previous_pool = target.pool
        module.clear_collective_graphs(SimpleNamespace(model_runner=runner))
        assert (
            target.pool is not previous_pool
            and target.pool is Platform._global_graph_pool
        )
        assert all(
            m.pool is target.pool
            and not m.graphs
            and not m._graphs_captured
            and m._capture_descs
            for m in (target, draft, inner)
        )
        assert wrapper.graph_pool is target.pool
        assert runner.model is weights and runner.kv_caches is kv
        assert encoder.inputs_embeds is embeddings and encoder.encoder_cache is cache
    assert events.count("encoder-clear") == 2


def worker_methods(scope):
    path = ROOT / "vllm/v1/worker/gpu_worker.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "Worker"
    )
    names = {"network_sleep_status", "sleep_network", "wake_network"}
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
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
    module, ps, _ = context
    ps.get_world_group = lambda: SimpleNamespace(rank=15)
    events = []
    groups = {"tp:0": object()}
    module.check_network_sleep = lambda worker: events.append("check")
    module.live_groups = lambda: groups
    module.clear_collective_graphs = lambda worker: events.append("graphs-released")
    module.close_transport = lambda original: events.append(("close", original))
    module.restore_group_references = lambda original: events.append(
        ("restore", original)
    )
    monkeypatch.setitem(sys.modules, "vllm.distributed.network_sleep", module)
    worker = worker_methods(
        {
            "torch": SimpleNamespace(
                accelerator=SimpleNamespace(
                    synchronize=lambda: events.append("synchronize")
                )
            ),
            "set_current_vllm_config": lambda config: nullcontext(),
            "init_worker_distributed_environment": lambda *args, **kwargs: (
                events.append(("init", kwargs["network_wake_generation"]))
            ),
            "current_platform": SimpleNamespace(dist_backend="nccl"),
        }
    )
    worker.rank, worker.local_rank, worker.distributed_init_method = (
        3,
        0,
        "tcp://host:1",
    )
    worker.synchronize_device = lambda: events.append("synchronize")
    worker.vllm_config = object()
    worker.model_runner = SimpleNamespace(
        model=object(),
        kv_cache=object(),
        capture_model=lambda: events.append("capture"),
    )
    model, cache = worker.model_runner.model, worker.model_runner.kv_cache
    # Current main exposes synchronize_device via WorkerBase.
    for generation in (1, 2):
        events.clear()
        slept = worker.sleep_network()
        assert slept == {
            "state": "sleeping",
            "rank": 15,
            "pid": os.getpid(),
            "generation": generation,
        }
        assert worker.sleep_network() == slept
        assert events == ["check", "synchronize", "graphs-released", ("close", groups)]
        events.clear()
        woke = worker.wake_network()
        assert woke["state"] == "active" and woke["generation"] == generation
        assert worker.wake_network() == woke
        assert events == [
            ("init", generation),
            ("restore", groups),
            "capture",
            "synchronize",
        ]
        assert (
            worker.model_runner.model is model and worker.model_runner.kv_cache is cache
        )


def test_engine_control_lifecycle_order_and_idempotence(context, monkeypatch):
    module, _, _ = context
    events = []
    engine = SimpleNamespace(
        model_executor=SimpleNamespace(
            collective_rpc=lambda method: events.append(method)
        ),
        pause_scheduler=lambda **kwargs: None,
        resume_scheduler=lambda: events.append("resume"),
    )
    monkeypatch.setattr(
        module, "close_engine_control_group", lambda e: events.append("control-close")
    )
    monkeypatch.setattr(
        module, "restore_engine_control_group", lambda e: events.append("control-open")
    )
    for _ in range(2):
        events.clear()
        core_method("sleep")(engine, level=0, mode="wait")
        core_method("sleep")(engine, level=0, mode="wait")
        assert events == ["check_network_sleep", "sleep_network", "control-close"]
        assert engine._network_sleep_state == "sleeping"
        core_method("wake_up")(engine, tags=["scheduling"])
        core_method("wake_up")(engine, tags=["scheduling"])
        assert events[-3:] == ["control-open", "wake_network", "resume"]
        assert engine._network_sleep_state == "active"


def test_failed_control_close_leaves_engine_closed_to_retry(context, monkeypatch):
    module, _, _ = context
    events = []

    def fail(engine):
        raise RuntimeError("control close failed")

    monkeypatch.setattr(module, "close_engine_control_group", fail)
    engine = SimpleNamespace(
        model_executor=SimpleNamespace(
            collective_rpc=lambda method: events.append(method)
        ),
        pause_scheduler=lambda **kwargs: None,
    )
    with pytest.raises(RuntimeError, match="control close failed"):
        core_method("sleep")(engine, level=0, mode="wait")
    assert engine._network_sleep_state == "closing"
    with pytest.raises(RuntimeError, match="did not finish"):
        core_method("wake_up")(engine)
    assert events == ["check_network_sleep", "sleep_network"]


def test_restore_rebinds_ags_manager_to_original_group_objects(context):
    module, ps, _ = context
    old_tp, old_dp = put(ps, "tp:0"), put(ps, "dp:0")
    fresh_tp, fresh_dp = put(ps, "tp:0"), put(ps, "dp:0")
    manager = SimpleNamespace(tp_group=fresh_tp, dp_group=fresh_dp)
    ep = put(ps, "ep:0", transport=SimpleNamespace(all2all_manager=manager))
    old_ep = ps.GroupCoordinator("ep:0", [0, 1], None)
    ps._TP, ps._DP, ps._EP = fresh_tp, fresh_dp, ep
    module.restore_group_references({"tp:0": old_tp, "dp:0": old_dp, "ep:0": old_ep})
    assert ps._EP.device_communicator.all2all_manager is manager
    assert manager.tp_group is old_tp and manager.dp_group is old_dp


def test_actual_dp_busy_loop_pauses_before_control_close_and_never_uses_closed_group(
    context, monkeypatch
):
    """Execute real pause/callback/DP loop code through consensus and sleeping idle."""
    import queue
    from functools import partial
    from typing import Literal, get_args

    module, _, _ = context
    path = ROOT / "vllm/v1/engine/core.py"
    tree = ast.parse(path.read_text())
    selections = {
        "EngineCore": {"sleep", "_finish_pause"},
        "EngineCoreProc": {
            "pause_scheduler",
            "has_work",
            "_process_input_queue",
            "_notify_idle_state_callbacks",
        },
        "DPEngineCoreProc": {
            "_pause_complete",
            "_has_global_unfinished_reqs",
            "run_busy_loop",
        },
    }
    methods = []
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name in selections:
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name in selections[cls.name]:
                    fn.decorator_list = []
                    methods.append(fn)
    extracted = ast.ClassDef(
        name="Engine", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    prefix = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    events = []
    group = object()

    def sync(group_arg, **kw):
        assert group_arg is group
        assert kw == {"has_unfinished": False, "pending_pause": True}
        events.append("consensus")
        return False, True

    scope = dict(
        Any=object,
        Future=Future,
        partial=partial,
        get_args=get_args,
        PauseMode=Literal["abort", "keep", "wait"],
        PauseState=SimpleNamespace(PAUSED_ALL=2, PAUSED_NEW=1),
        envs=SimpleNamespace(VLLM_SLEEP_RELEASE_TRANSPORT=True),
        queue=queue,
        DEBUG=10,
        ParallelConfig=SimpleNamespace(sync_dp_state=sync),
        EngineCoreOutputs=lambda **kw: SimpleNamespace(**kw),
        logger=SimpleNamespace(
            info=lambda *a: None, debug=lambda *a: None, isEnabledFor=lambda _: False
        ),
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[prefix, extracted], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        scope,
    )
    engine = scope["Engine"]()
    engine.model_executor = SimpleNamespace(
        is_sleeping=False, collective_rpc=lambda m: events.append(m)
    )
    engine.scheduler = SimpleNamespace(
        set_pause_state=lambda state: None,
        has_requests=lambda: False,
        has_unfinished_requests=lambda: False,
    )
    engine.batch_queue = None
    engine._idle_state_callbacks = []
    engine.pending_pause, engine.engines_running, engine.ignore_start_dp_wave = (
        False,
        False,
        False,
    )
    engine.dp_rank, engine.dp_size, engine.dp_group = 0, 4, group
    engine.dp_sync_interval, engine.step_counter, engine.current_wave = 32, 0, 0
    engine.input_queue, engine.output_queue, engine.aborts_queue = (
        queue.Queue(),
        queue.Queue(),
        queue.Queue(),
    )
    engine.process_input_queue_block = False
    engine.is_running = lambda: True
    engine._maybe_publish_request_counts = lambda: None
    engine._process_engine_step = lambda: False
    engine.eep_scaling_state = None
    engine.has_coordinator = True
    engine.capture_iteration_details = lambda _: nullcontext(None)
    engine.execute_dummy_batch = lambda: events.append("dummy")
    remaining = iter([True, True, True, False])
    engine._handle_shutdown = lambda: next(remaining)
    utils = ModuleType("vllm.distributed.utils")
    utils.stateless_destroy_torch_distributed_process_group = lambda g: events.append(
        "control-close"
    )
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    result = engine.sleep(level=0, mode="wait")
    assert not result.done() and engine.pending_pause and engine.engines_running
    with pytest.raises(SystemExit):
        engine.run_busy_loop()
    assert result.done() and result.exception() is None
    assert events == [
        "check_network_sleep",
        "dummy",
        "consensus",
        "synchronize_device",
        "sleep_network",
        "control-close",
    ]
    assert engine.dp_group is None and not engine.engines_running
    assert engine.ignore_start_dp_wave and engine._network_sleep_state == "sleeping"
