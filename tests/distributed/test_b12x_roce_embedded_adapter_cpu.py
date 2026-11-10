# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RoCEnante's binding to standalone and FlashInfer-embedded b12x, without a GPU.

The runtimes here are fakes that record what the adapter asks of them; the
native gate on a Spark allocation establishes transport and numerics.
"""

import ast
import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from vllm.distributed.device_communicators import (
    b12x_roce_all_reduce as adapter_module,
)

MB = 1024 * 1024
BF16, F16, F32 = torch.bfloat16, torch.float16, torch.float32
NAMES = {F16: "float16", BF16: "bfloat16", F32: "float32"}
LIVE = (
    "RoCEnante all-reduce is live: first routed all-reduce is 245760 bytes "
    "(bfloat16); NCCL remains the fallback above 2MB."
)


def _nbytes(tensor):
    return tensor.numel() * tensor.element_size()


class _Fabric:
    """What every fake shares: the call record and the capture state."""

    def __init__(self):
        self.events = []
        self.runtimes = []
        self.capturing = False

    def named(self, kind):
        return [event for event in self.events if event[1] == kind]

    def launches(self):
        return [(event[0], *event[2:]) for event in self.named("launch")]


class _LauncherModule:
    """A `_oneshot_cute`/`_allgather_cute` stand-in: one launcher per key."""

    def __init__(self, fabric, kind):
        self.fabric, self.kind, self.cache = fabric, kind, {}

    def get_launcher(self, *key):
        if key not in self.cache:
            assert not self.fabric.capturing, "launcher compiled during capture"
            self.fabric.events.append((self.kind, "compile", key))

            def launch(runtime, inp):
                label = key[0] if self.kind == "reduce" else self.kind
                state = self.fabric.capturing
                self.fabric.events.append(
                    (runtime, "launch", label, _nbytes(inp), state)
                )

            self.cache[key] = launch
        return self.cache[key]


class _Runtime:
    """The surface both b12x flavours share."""

    def __init__(
        self, *, exchange_group, device, max_size=2 * MB, max_gather_bytes=16 * MB
    ):
        self.fabric, self.name = exchange_group.fabric, exchange_group.name
        self.world_size, self.rank = exchange_group.world, exchange_group.rank
        self.max_size, self.max_gather_bytes = max_size, max_gather_bytes
        self.hca_names = ("rocep1s0f0", "roceP2p1s0f0")
        self.scratch = self.closed = False
        self.failure = None
        self.fabric.runtimes.append(self)

    @classmethod
    def from_exchange_group(cls, **kwargs):
        return cls(**kwargs)

    def should_allreduce(self, inp):
        size = _nbytes(inp)
        return inp.dtype in NAMES and 0 < size <= self.max_size and size % 16 == 0

    def should_all_gather(self, inp, dim=-1):
        dim = dim + inp.dim() if dim < 0 else dim
        return dim in (0, inp.dim() - 1) and 0 < _nbytes(inp) <= self.max_gather_bytes

    @contextmanager
    def capture(self, stream=None, *, channel_id=None):
        self.fabric.events.append((self.name, "capture:enter", stream))
        try:
            yield self
        finally:
            self.fabric.events.append((self.name, "capture:exit", stream))

    def check_health(self):
        if self.failure:
            raise RuntimeError(self.failure)

    def close(self):
        self.closed = True


class StandaloneAllReduce(_Runtime):
    """Standalone b12x 1.2.6: the public collectives plan internally."""

    def prepare(self, dtypes=(BF16,), *, padded_gather=False):
        self.fabric.events.append((self.name, "prepare", tuple(dtypes)))

    def all_reduce(
        self, inp, *, out=None, stream=None, channel_id=None, peer_input_ptrs=None
    ):
        self.fabric.events.append((self.name, "all_reduce", NAMES[inp.dtype]))
        return inp * self.world_size

    def all_gather(self, inp, *, dim=-1, out=None, stream=None):
        self.fabric.events.append((self.name, "all_gather", dim))
        return torch.cat([inp] * self.world_size, dim=dim)


class RoceOneshotAllReduce(_Runtime):
    """Embedded b12x 1.5.0: the public collectives run a published plan."""

    def _launcher_key(self, dtype):
        return (NAMES[dtype], self.world_size, self.rank, 512, 2, 64, 2, 0)

    def _gather_launcher_key(self):
        return self._launcher_key(BF16)[1:]

    def _prepare_resources(self, dtypes=(BF16,), *, padded_gather=False):
        assert not self.fabric.capturing, "resources prepared during capture"
        self.fabric.events.append(
            (self.name, "resources", tuple(dtypes), padded_gather)
        )
        for dtype in dtypes:
            self.fabric.reduce.get_launcher(*self._launcher_key(dtype))
        self.fabric.gather.get_launcher(*self._gather_launcher_key())
        self.scratch = True

    def all_reduce(
        self,
        inp,
        *,
        out=None,
        stream=None,
        channel_id=None,
        peer_input_ptrs=None,
        plan,
    ):
        raise TypeError(f"comm.roce requires a Plan, got {type(plan).__name__}")

    def all_gather(self, inp, *, dim=-1, out=None, stream=None, plan):
        raise TypeError(f"comm.roce requires a Plan, got {type(plan).__name__}")

    def _run_prepared_all_reduce(self, inp, *, prepared, out=None, stream=None):
        assert prepared.runtime is self and self.scratch
        if not self.should_allreduce(inp):
            raise ValueError("input is not eligible for the RoCE one-shot all-reduce")
        if inp.dtype not in prepared.reduce_launchers:
            raise RuntimeError("RoCE all-reduce dtype was not prepared")
        prepared.reduce_launchers[inp.dtype](self.name, inp)
        return inp * self.world_size

    def _run_prepared_all_gather(self, inp, *, prepared, dim=-1, out=None, stream=None):
        assert prepared.runtime is self
        if not self.should_all_gather(inp, dim):
            raise ValueError("input is not eligible for the RoCE all-gather")
        prepared.gather_launcher(self.name, inp)
        return torch.cat([inp] * self.world_size, dim=dim)


class SessionOnlyAllReduce(_Runtime):
    """A plan-taking runtime without the launch path this adapter binds."""

    def all_reduce(self, inp, *, out=None, stream=None, plan):
        raise AssertionError("never constructed")


def _install(monkeypatch, runtime, *, world=8, rank=0, hierarchical_min="0"):
    fabric = _Fabric()
    roce = types.ModuleType("b12x.comm.roce")
    roce.API_VERSION, roce.AllReduce = 1, runtime
    roce.is_supported = lambda device: True
    if runtime is RoceOneshotAllReduce:
        fabric.reduce = roce._oneshot_cute = _LauncherModule(fabric, "reduce")
        fabric.gather = roce._allgather_cute = _LauncherModule(fabric, "gather")
        # The session-prepared public API must not be used.
        roce.plan = roce.query_from_runtime = lambda *a, **k: pytest.fail("session")
    comm = types.ModuleType("b12x.comm")
    comm.roce = roce
    root = types.ModuleType("b12x")
    root.comm = comm
    for name, module in (("b12x", root), ("b12x.comm", comm), (roce.__name__, roce)):
        monkeypatch.setitem(sys.modules, name, module)

    def group(name, members):
        position = members.index(rank) if rank in members else -1
        return types.SimpleNamespace(
            fabric=fabric, name=name, world=len(members), rank=position
        )

    def new_group(ranks, **kwargs):
        return group("row" if ranks[1] - ranks[0] == 1 else "column", ranks)

    def gather_object(results, local, group):
        results[:] = [local] * len(results)

    dist = types.SimpleNamespace(
        get_rank=lambda group: group.rank,
        get_world_size=lambda group: group.world,
        get_process_group_ranks=lambda group: list(range(group.world)),
        all_gather_object=gather_object,
        new_group=new_group,
        destroy_process_group=Mock(),
    )
    envs = types.SimpleNamespace(
        VLLM_ROCE_ALLREDUCE_MAX_SIZE="2MB",
        VLLM_ROCE_ALLGATHER_MAX_SIZE="2MB",
        VLLM_ROCE_HIERARCHICAL_ALLREDUCE=hierarchical_min != "0",
        VLLM_ROCE_HIERARCHICAL_MIN_SIZE=hierarchical_min,
        # The SM12x fused collectives need integrator hooks the embedded
        # runtime does not have; these tests cover the plain collectives.
        VLLM_ROCE_FUSED_COLLECTIVES=False,
    )
    monkeypatch.setattr(adapter_module, "dist", dist)
    monkeypatch.setattr(adapter_module, "envs", envs)
    monkeypatch.setattr(adapter_module, "logger", Mock())
    monkeypatch.setattr(
        adapter_module,
        "in_the_same_node_as",
        lambda group, source_rank=0: [True] + [False] * (world - 1),
    )
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: fabric.capturing
    )
    adapter = adapter_module.B12xRoceAllReduce(
        group("tp", list(range(world))), object(), torch.device("cpu")
    )
    return adapter, fabric


@contextmanager
def _graph(adapter, fabric, stream=None):
    """`graph_capture()` followed by `torch.cuda.graph()` on its stream."""
    with adapter.capture(stream=stream):
        fabric.capturing = True
        try:
            yield
        finally:
            fabric.capturing = False


def _messages(method):
    return [call.args[0] % call.args[1:] for call in method.call_args_list]


def test_first_forward_all_reduce_runs_on_embedded_b12x(monkeypatch):
    """2026-11-12, TP8: `all_reduce() missing 1 required keyword-only argument`."""
    adapter, fabric = _install(monkeypatch, RoceOneshotAllReduce)
    assert not adapter.disabled
    hidden = torch.ones((24, 5120), dtype=BF16)
    assert adapter.should_custom_ar(hidden)
    assert torch.equal(adapter.custom_all_reduce(hidden), hidden * 8)
    assert LIVE in _messages(adapter_module.logger.info)
    assert fabric.launches() == [("tp", "bfloat16", 245760, False)]
    # The launcher is the one the runtime's own key names, resolved once.
    key = ("bfloat16", 8, 0, 512, 2, 64, 2, 0)
    assert fabric.named("compile") == [
        ("reduce", "compile", key),
        ("gather", "compile", key[1:]),
    ]


def test_standalone_b12x_is_used_unchanged(monkeypatch):
    adapter, fabric = _install(monkeypatch, StandaloneAllReduce)
    assert type(adapter._runtime) is StandaloneAllReduce
    hidden, stream = torch.ones((24, 5120), dtype=BF16), object()
    assert torch.equal(adapter.custom_all_reduce(hidden), hidden * 8)
    assert adapter.should_all_gather(hidden, 1)
    assert adapter.all_gather(hidden, 1).shape == (24, 8 * 5120)
    with adapter.capture(stream=stream):
        pass
    assert fabric.events == [
        ("tp", "all_reduce", "bfloat16"),
        ("tp", "all_gather", 1),
        ("tp", "prepare", (BF16, F16, F32)),
        ("tp", "capture:enter", stream),
        ("tp", "capture:exit", stream),
    ]
    assert LIVE in _messages(adapter_module.logger.info)


def test_capture_prepares_before_the_graph_and_nothing_during_it(monkeypatch):
    adapter, fabric = _install(monkeypatch, RoceOneshotAllReduce)
    with _graph(adapter, fabric):
        entered = len(fabric.events)
        for dtype in (BF16, F16, F32):
            for shape in ((1, 7168), (9, 7168), (24, 5120)):
                value = torch.ones(shape, dtype=dtype)
                assert adapter.should_custom_ar(value)
                assert torch.equal(adapter.custom_all_reduce(value), value * 8)
        shard = torch.ones((9, 224), dtype=BF16)
        assert adapter.should_all_gather(shard, 1)
        assert adapter.all_gather(shard, 1).shape == (9, 8 * 224)
    before, during = fabric.events[:entered], fabric.events[entered:-1]
    assert [event[1] for event in before] == [
        "resources",
        *["compile"] * 4,
        "capture:enter",
    ]
    assert before[0] == ("tp", "resources", (BF16, F16, F32), False)
    assert len(during) == 10 and all(
        event[1] == "launch" and event[-1] for event in during
    )


def test_one_launcher_serves_every_size_and_outlives_the_library_cache(monkeypatch):
    adapter, fabric = _install(monkeypatch, RoceOneshotAllReduce)
    sizes = [(1, 8), (1, 7168), (24, 5120), (128, 8192)]
    for shape in sizes:
        adapter.custom_all_reduce(torch.ones(shape, dtype=BF16))
    # What the end of any b12x preparation job does to programs it does not own.
    fabric.reduce.cache.clear()
    fabric.gather.cache.clear()
    for _ in range(2):
        with _graph(adapter, fabric):
            for shape in sizes:
                adapter.custom_all_reduce(torch.ones(shape, dtype=BF16))
            adapter.all_gather(torch.ones((9, 224), dtype=BF16), 1)
    compiled = [event[2][0] for event in fabric.named("compile")]
    assert compiled == ["bfloat16", 8, "float16", "float32"]
    assert len(fabric.named("resources")) == 1
    assert len(fabric.launches()) == 4 + 2 * 5


def test_dtype_not_prepared_before_a_capture_goes_to_nccl(monkeypatch):
    from vllm.distributed.device_communicators import cuda_communicator

    adapter, fabric = _install(monkeypatch, RoceOneshotAllReduce)
    monkeypatch.setattr(
        cuda_communicator, "should_nccl_symm_mem_allreduce", lambda *args: False
    )
    nccl = Mock(disabled=False, world_size=8)
    nccl.all_reduce.side_effect = lambda value: value * 8
    communicator = types.SimpleNamespace(
        b12x_ar_comm=adapter,
        pynccl_comm=nccl,
        fi_ar_comm=None,
        qr_comm=None,
        fi_pcie_ipc_ar_comm=None,
        aiter_ar_comm=None,
        use_aiter_allreduce=False,
        ca_comm=None,
        symm_mem_comm=None,
    )
    all_reduce = cuda_communicator.CudaCommunicator.all_reduce
    value = torch.ones((9, 7168), dtype=BF16)

    fabric.capturing = True  # a capture the adapter was not told about
    assert not adapter.should_custom_ar(value)
    assert adapter.custom_all_reduce(value) is None
    assert not adapter.should_all_gather(value, 1)
    assert torch.equal(all_reduce(communicator, value), value * 8)
    assert nccl.all_reduce.call_count == 1 and fabric.events == []

    fabric.capturing = False  # eager: prepared on first use, as standalone did
    assert torch.equal(all_reduce(communicator, value), value * 8)
    assert fabric.launches() == [("tp", "bfloat16", 129024, False)]

    fabric.capturing = True
    assert adapter.should_custom_ar(value) and adapter.should_all_gather(value, 1)
    assert not adapter.should_custom_ar(value.float())
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        adapter._runtime.prepare((F32,))
    assert nccl.all_reduce.call_count == 1


def test_size_limit_and_ineligible_tensors_stay_on_nccl(monkeypatch):
    adapter, fabric = _install(monkeypatch, RoceOneshotAllReduce)
    for value in (
        torch.ones((129, 8192), dtype=BF16),  # 16 bytes above 2MB
        torch.ones((3, 3), dtype=BF16),  # not a multiple of 16 bytes
        torch.ones((4, 4), dtype=torch.int32),
    ):
        assert not adapter.should_custom_ar(value)
        assert adapter.custom_all_reduce(value) is None
    assert not adapter.should_all_gather(torch.ones((129, 8192), dtype=BF16), 1)
    assert fabric.events == []


def test_all_gather_runs_the_prepared_gather_launcher(monkeypatch):
    adapter, fabric = _install(monkeypatch, RoceOneshotAllReduce)
    shard = torch.arange(9 * 224, dtype=F32).reshape(9, 224)
    assert adapter.should_all_gather(shard, 1) and adapter.should_all_gather(shard, 0)
    assert torch.equal(adapter.all_gather(shard, 1), torch.cat([shard] * 8, dim=1))
    assert adapter.all_gather(shard, 0).shape == (72, 224)
    assert fabric.launches() == [("tp", "gather", 8064, False)] * 2
    assert [event[0] for event in fabric.named("compile")] == ["gather"]
    assert "RoCEnante all-gather is live" in _messages(adapter_module.logger.info)[-1]


def test_driver_keeps_the_standalone_prepare_and_attributes(monkeypatch):
    """What the native gates reach through `adapter._runtime`."""
    adapter, fabric = _install(monkeypatch, RoceOneshotAllReduce)
    adapter._runtime.prepare((BF16, F32), padded_gather=True)
    adapter._runtime.prepare((BF16, F32))
    assert fabric.named("resources") == [("tp", "resources", (BF16, F32), True)]
    assert adapter._runtime.max_size == adapter._runtime.max_gather_bytes == 2 * MB
    assert adapter._runtime.hca_names == ("rocep1s0f0", "roceP2p1s0f0")
    assert not hasattr(adapter._runtime, "prepare_graph_all_reduce")


def test_hierarchy_routes_bf16_through_prepared_fp32_subgroups(monkeypatch):
    adapter, fabric = _install(
        monkeypatch, RoceOneshotAllReduce, world=16, rank=6, hierarchical_min="64KB"
    )
    assert [runtime.name for runtime in fabric.runtimes] == ["tp", "row", "column"]
    # Each four-rank runtime resolved its FP32 launcher at initialization.
    assert [event[2][:3] for event in fabric.named("compile")] == [
        ("float32", 4, 2),
        (4, 2, 512),
        ("float32", 4, 1),
        (4, 1, 512),
    ]
    below = torch.ones((1, 32760), dtype=BF16)  # 16 bytes under the 64KB minimum
    above = torch.ones((9, 7168), dtype=BF16)
    stream = object()
    with _graph(adapter, fabric, stream):
        entered = len(fabric.events)
        for value in (below, above):
            reduced = adapter.custom_all_reduce(value)
            assert reduced.dtype == BF16 and torch.equal(reduced, value * 16)
    assert fabric.launches() == [
        ("tp", "bfloat16", 65520, True),
        ("row", "float32", 258048, True),
        ("column", "float32", 258048, True),
    ]
    assert not [e for e in fabric.events[entered:] if e[1] in ("compile", "resources")]
    assert [event[0] for event in fabric.named("capture:enter")] == [
        "tp",
        "row",
        "column",
    ]
    assert all(event[2] is stream for event in fabric.named("capture:enter"))
    assert [event[0] for event in fabric.named("capture:exit")] == [
        "column",
        "row",
        "tp",
    ]


def test_health_and_close_reach_every_embedded_runtime(monkeypatch):
    adapter, fabric = _install(
        monkeypatch, RoceOneshotAllReduce, world=16, rank=5, hierarchical_min="64KB"
    )
    adapter.check_health()
    fabric.runtimes[2].failure = "RoCE collective on rank 1 timed out"
    with pytest.raises(RuntimeError, match="timed out"):
        adapter.check_health()
    adapter.close()
    assert all(runtime.closed for runtime in fabric.runtimes)
    assert adapter.disabled and adapter._runtime is None


def test_embedded_runtime_without_the_launch_path_disables_every_rank(monkeypatch):
    adapter, fabric = _install(monkeypatch, SessionOnlyAllReduce)
    assert adapter.disabled and adapter._runtime is None and not fabric.runtimes
    (warning,) = _messages(adapter_module.logger.warning)
    assert "embedded b12x.comm.roce lacks _prepare_resources" in warning
    assert "_run_prepared_all_gather" in warning and "launcher modules" in warning


def test_installed_flashinfer_keeps_the_bound_entry_points():
    """The private surface `_EmbeddedRoce` binds, read from the installed source."""
    spec = importlib.util.find_spec("flashinfer")
    root = Path(spec.origin).parent if spec and spec.origin else Path()
    roce = root / "experimental" / "b12x" / "comm" / "roce"
    if not (roce / "roce_oneshot.py").exists():
        pytest.skip("flashinfer-python does not embed the b12x RoCE runtime")

    def functions(path, owner=None):
        body = ast.parse(path.read_text()).body
        if owner is not None:
            body = next(
                node.body
                for node in body
                if isinstance(node, ast.ClassDef) and node.name == owner
            )
        return {node.name: node for node in body if isinstance(node, ast.FunctionDef)}

    def names(function):
        return [arg.arg for arg in function.args.args + function.args.kwonlyargs]

    runtime = functions(roce / "roce_oneshot.py", "RoceOneshotAllReduce")
    if "plan" not in names(runtime["all_reduce"]):
        pytest.skip("the embedded runtime still plans internally")
    assert "plan" in names(runtime["all_gather"])
    assert set(adapter_module._EmbeddedRoce.ENTRY_POINTS) <= set(runtime)
    assert names(runtime["_prepare_resources"]) == ["self", "dtypes", "padded_gather"]
    assert names(runtime["_run_prepared_all_reduce"]) == [
        "self",
        "inp",
        "prepared",
        "out",
        "stream",
    ]
    assert names(runtime["_run_prepared_all_gather"]) == [
        "self",
        "inp",
        "prepared",
        "dim",
        "out",
        "stream",
    ]
    # The launch path reads only what the driver's state object carries.
    read = {
        node.attr
        for function in runtime.values()
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "prepared"
    }
    assert read == {"runtime", "reduce_launchers", "gather_launcher"}
    # Each key method returns exactly its launcher factory's arguments.
    for module, key in (
        ("_oneshot_cute.py", "_launcher_key"),
        ("_allgather_cute.py", "_gather_launcher_key"),
    ):
        (returned,) = [
            node.value
            for node in ast.walk(runtime[key])
            if isinstance(node, ast.Return)
        ]
        factory = functions(roce / module)["get_launcher"]
        assert len(returned.elts) == len(names(factory))
