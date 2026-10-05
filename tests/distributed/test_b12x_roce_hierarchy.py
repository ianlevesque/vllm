# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU source-method checks for hierarchy dispatch and fail-stop orchestration.

No CUDA/native vLLM imports; exact-image distributed gates establish numerics.
"""

import ast
import os
import sys
from contextlib import ExitStack, contextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


class Tensor:
    def __init__(self, size, dtype, history=()):
        self.size, self.dtype, self.history = size, dtype, history

    def numel(self):
        return self.size // (4 if self.dtype == "fp32" else 2)

    def element_size(self):
        return 4 if self.dtype == "fp32" else 2

    def float(self):
        return Tensor(self.numel() * 4, "fp32", self.history + ("float",))

    def to(self, dtype):
        return Tensor(self.numel() * 2, dtype, self.history + ("cast",))


@pytest.fixture
def adapter():
    root = Path(os.environ.get("VLLM_SOURCE_ROOT", Path(__file__).resolve().parents[2]))
    source = root / "vllm/distributed/device_communicators/b12x_roce_all_reduce.py"
    node = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "B12xRoceAllReduce"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    dist = Mock()
    torch = SimpleNamespace(bfloat16="bf16", float32="fp32", float16="fp16")
    namespace = dict(
        torch=torch,
        dist=dist,
        ExitStack=ExitStack,
        contextmanager=contextmanager,
        timedelta=timedelta,
        logger=Mock(),
        envs=SimpleNamespace(VLLM_ROCE_ALLREDUCE_MAX_SIZE="2MB"),
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    value = namespace["B12xRoceAllReduce"].__new__(namespace["B12xRoceAllReduce"])
    value.disabled = False
    value.rank, value.world_size = 0, 16
    value.group, value.device = "tp16", "cuda:0"
    value._runtime = Mock()
    value._runtime.should_allreduce.return_value = True
    value._hierarchical_runtimes = [Mock(), Mock()]
    value._hierarchical_groups = ["row", "column"]
    value._hierarchical_min_size = 65536
    value._hierarchical_max_size = 1048576
    value._announced = value._announced_hierarchy = False
    return value, dist


@pytest.mark.parametrize(
    "size,dtype,expected",
    [
        (65520, "bf16", False),
        (65536, "bf16", True),
        (1048576, "bf16", True),
        (1048592, "bf16", False),
        (129024, "fp16", False),
        (129024, "fp32", False),
    ],
)
def test_dispatch_bounds_and_precision(adapter, size, dtype, expected):
    value, _ = adapter
    assert value._should_hierarchical(Tensor(size, dtype)) is expected


def test_hierarchy_retains_fp32_between_stages(adapter):
    value, _ = adapter

    def reduce(tensor):
        assert tensor.dtype == "fp32"
        return Tensor(tensor.size, tensor.dtype, tensor.history + ("reduce",))

    for runtime in value._hierarchical_runtimes:
        runtime.all_reduce.side_effect = reduce
    result = value.custom_all_reduce(Tensor(129024, "bf16"))
    assert result.dtype == "bf16"
    assert result.history == ("float", "reduce", "reduce", "cast")
    value._runtime.all_reduce.assert_not_called()


def test_ineligible_reduction_uses_original_runtime(adapter):
    value, _ = adapter
    tensor = Tensor(14336, "bf16")
    value.custom_all_reduce(tensor)
    value._runtime.all_reduce.assert_called_once_with(tensor)
    for runtime in value._hierarchical_runtimes:
        runtime.all_reduce.assert_not_called()


def test_configuration_disagreement_disables_every_rank(adapter):
    value, dist = adapter

    def votes(result, local, group):
        result[:] = [(None, (2097152, 2097152, 65536))] * 15 + [
            (None, (2097152, 2097152, 0))
        ]

    dist.all_gather_object.side_effect = votes
    assert "size limits differ" in value._exchange_vote(None, (2097152, 2097152, 65536))


@pytest.mark.parametrize("stage", [0, 1])
def test_each_subgroup_failure_is_fail_stop(adapter, stage):
    value, _ = adapter
    value._hierarchical_runtimes[stage].check_health.side_effect = RuntimeError(
        "poisoned"
    )
    with pytest.raises(RuntimeError, match="poisoned"):
        value.check_health()


def test_close_releases_every_runtime_and_subgroup(adapter):
    value, dist = adapter
    runtimes = [value._runtime, *value._hierarchical_runtimes]
    value.close()
    for runtime in runtimes:
        runtime.close.assert_called_once_with()
    assert [call.args[0] for call in dist.destroy_process_group.call_args_list] == [
        "row",
        "column",
    ]
    assert value.disabled and value._runtime is None
    assert not value._hierarchical_runtimes and not value._hierarchical_groups


@pytest.mark.parametrize("global_base", [0, 16])
def test_subgroups_respect_tp_rank_mapping(adapter, monkeypatch, global_base):
    value, dist = adapter
    value.rank = 7
    value._hierarchical_runtimes.clear()
    value._hierarchical_groups.clear()
    dist.get_process_group_ranks.return_value = list(
        range(global_base, global_base + 16)
    )
    dist.new_group.side_effect = lambda **kwargs: tuple(kwargs["ranks"])
    dist.all_gather_object.side_effect = lambda results, local, group: (
        results.__setitem__(slice(None), [local] * 16)
    )
    roce = SimpleNamespace(AllReduce=Mock(side_effect=lambda **kwargs: Mock()))
    monkeypatch.setitem(sys.modules, "b12x.comm", SimpleNamespace(roce=roce))
    value._initialize_hierarchy(2097152, 65536)
    assert value._hierarchical_groups == [
        tuple(global_base + x for x in (4, 5, 6, 7)),
        tuple(global_base + x for x in (3, 7, 11, 15)),
    ]
    assert len(dist.new_group.call_args_list) == 8
    assert all(
        call.kwargs["use_local_synchronization"]
        for call in dist.new_group.call_args_list
    )
    assert len(value._hierarchical_runtimes) == 2
    assert value._hierarchical_max_size == 1048576


def test_peer_initialization_failure_closes_partial_hierarchy(adapter, monkeypatch):
    value, dist = adapter
    value._hierarchical_runtimes.clear()
    value._hierarchical_groups.clear()
    dist.get_process_group_ranks.return_value = list(range(16))
    dist.new_group.side_effect = lambda **kwargs: tuple(kwargs["ranks"])

    def peer_failure(results, local, group):
        results[:] = [None] * 15 + ["peer preparation failed"]

    dist.all_gather_object.side_effect = peer_failure
    runtime = Mock()
    roce = SimpleNamespace(AllReduce=Mock(return_value=runtime))
    monkeypatch.setitem(sys.modules, "b12x.comm", SimpleNamespace(roce=roce))
    with pytest.raises(RuntimeError, match="peer preparation failed"):
        value._initialize_hierarchy(2097152, 65536)
    runtime.close.assert_called_once_with()
    assert value.disabled and not value._hierarchical_runtimes
    assert len(dist.destroy_process_group.call_args_list) == 2


def test_capture_uses_one_stream_and_unwinds_every_runtime(adapter):
    value, _ = adapter
    events = []
    stream = object()

    @contextmanager
    def capture(name, **kwargs):
        assert kwargs["stream"] is stream
        events.append(name + ":enter")
        try:
            yield
        finally:
            events.append(name + ":exit")

    for name, runtime in zip(
        ("flat", "row", "column"), [value._runtime, *value._hierarchical_runtimes]
    ):
        runtime.capture.side_effect = lambda _name=name, **kwargs: capture(
            _name, **kwargs
        )
    with (
        pytest.raises(RuntimeError, match="capture body"),
        value.capture(stream=stream),
    ):
        raise RuntimeError("capture body")
    assert events == [
        "flat:enter",
        "row:enter",
        "column:enter",
        "column:exit",
        "row:exit",
        "flat:exit",
    ]
