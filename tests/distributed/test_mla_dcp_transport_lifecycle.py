# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU negative controls for the actual MLA cached-context gather implementation.

Native GPU collectives and loaded-model sleep remain separate qualifications.
"""

import ast
import functools
from pathlib import Path
from types import SimpleNamespace

import pytest


def manager_class(gather, direct_workspace=None):
    path = Path(__file__).parents[2] / "vllm/v1/attention/ops/dcp.py"
    body = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "MLADCPManager"
    )
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            body,
        ],
        type_ignores=[],
    )
    scope = {
        "functools": functools,
        "torch": SimpleNamespace(
            distributed=SimpleNamespace(all_gather_into_tensor=gather)
        ),
        "get_direct_dcp_kv_gather_workspace": lambda *args: direct_workspace,
        "logger": SimpleNamespace(info_once=lambda *args: None),
    }
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), scope)
    return scope["MLADCPManager"]


def make_manager(cls, world_size):
    manager = cls.__new__(cls)
    manager.group = SimpleNamespace(world_size=world_size, device_group=object())
    manager.num_ubatches = 1
    workspace = SimpleNamespace(
        ndim=2,
        is_contiguous=lambda: True,
        shape=(8 * (world_size + 1), 576),
        device="cuda",
        dtype="bfloat16",
    )
    manager.init_kv_gather(workspace, 8 * world_size)
    return manager


@pytest.mark.parametrize("world_size", [4, 16])
def test_retained_mla_manager_uses_current_transport_for_each_wake(world_size):
    calls = []

    def gather(output, local, *, group):
        # A real retired ProcessGroupNCCL errors at this same call boundary.
        assert group is manager.group.device_group, "retired process group"
        calls.append((output, local, group))
        return "completed"

    manager = make_manager(manager_class(gather), world_size)
    output, local = object(), object()
    original_coordinator = manager.group
    for _ in range(3):
        expected = manager.group.device_group
        assert manager.kv_gather(output, local) == "completed"
        assert calls[-1] == (output, local, expected)
        manager.group.device_group = object()
        assert manager.group is original_coordinator
    assert len({id(call[2]) for call in calls}) == 3


def test_direct_workspace_keeps_its_implementation():
    calls = []

    def direct(output, local):
        calls.append((output, local))
        return "direct"

    def forbidden(*args, **kwargs):
        pytest.fail("direct workspace unexpectedly used torch fallback")

    manager = make_manager(manager_class(forbidden, SimpleNamespace(gather=direct)), 4)
    output, local = object(), object()
    assert manager.kv_gather(output, local) == "direct"
    assert calls == [(output, local)]
