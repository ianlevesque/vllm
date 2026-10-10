# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU execution of installed dummy-batch methods without importing GPU kernels."""

import ast
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[3]


def compile_nodes(path, nodes, scope):
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    scope.setdefault("__name__", __name__)
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *nodes], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        scope,
    )


@pytest.fixture
def installed():
    path = ROOT / "vllm/v1/worker/gpu/input_batch.py"
    tree = ast.parse(path.read_text())
    classes = [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name in ("InputBuffers", "InputBatch")
    ]
    scope = dict(
        np=np, torch=torch, dataclass=dataclass, random_uuid=lambda: str(uuid4())
    )
    compile_nodes(path, classes, scope)
    return scope


class Mode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = 3


@pytest.mark.parametrize(
    "num_reqs,query_len,padded",
    [
        (1, 1, 32),
        (1, 3, 32),
        (1, 4, 32),
        (5, 4, 32),
        (128, 4, 512),
        (512, 4, 2048),
        (1, 8, 64),
    ],
)
def test_idle_dp_dispatch_preserves_query_width_and_request_budget(
    installed, num_reqs, query_len, padded
):
    path = ROOT / "vllm/v1/worker/gpu/model_runner.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "GPUModelRunner"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "execute_model"
    )
    fn.decorator_list = []
    desc = SimpleNamespace(
        num_tokens=padded, num_reqs=None, max_query_len=None, cg_mode=Mode.PIECEWISE
    )
    sync = SimpleNamespace(uniform_token_count=query_len, num_reqs=max(num_reqs, 5))
    scope = dict(
        CUDAGraphMode=Mode,
        InputBatch=installed["InputBatch"],
        dispatch_cg_and_sync_dp=lambda *a, **kw: (desc, sync),
    )
    compile_nodes(path, [fn], scope)
    buffers = installed["InputBuffers"](num_reqs, padded, torch.device("cpu"))
    runner = SimpleNamespace(
        gather_batch_req_state=lambda *a: (None, 1),
        lora_config=None,
        is_encoder_decoder=False,
        cudagraph_manager=None,
        dp_size=4,
        dp_rank=1,
        parallel_config=None,
        ubatch_runner=None,
        decode_query_len=query_len,
        input_buffers=buffers,
        pcp_manager=None,
    )

    class Captured(Exception):
        pass

    captured = []

    def capture(batch, valid):
        captured.append(batch)
        raise Captured

    runner.prepare_dummy_attn = capture
    scheduler = SimpleNamespace(
        num_scheduled_tokens={str(i): 1 for i in range(num_reqs)},
        total_num_scheduled_tokens=num_reqs,
    )
    with pytest.raises(Captured):
        scope[fn.name](runner, scheduler, dummy_run=True)
    batch = captured[0]
    assert batch.num_reqs == num_reqs
    assert batch.num_tokens == num_reqs * query_len
    assert batch.num_tokens_after_padding == padded
    assert batch.num_scheduled_tokens.tolist() == [query_len] * num_reqs
    assert batch.query_start_loc_np.tolist() == list(
        range(0, num_reqs * query_len + 1, query_len)
    )
    assert (
        batch.input_ids.shape
        == batch.positions.shape
        == batch.is_padding.shape
        == (padded,)
    )
    assert batch.is_padding.all()
    path = ROOT / "vllm/v1/worker/gpu/dp_utils.py"
    sync_cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "DPSyncState"
    )
    scope = dict(torch=torch, dataclass=dataclass, replace=replace)
    compile_nodes(path, [sync_cls], scope)
    target_sync = scope["DPSyncState"](
        num_tokens_across_dp=torch.full((4,), padded, dtype=torch.int32),
        uniform_token_count=query_len,
        eager=False,
        num_reqs=max(num_reqs, 5),
    )
    path = ROOT / "vllm/v1/worker/gpu/spec_decode/speculator.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "DraftModelSpeculator"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_build_uniform_batch_dp_sync"
    )
    compile_nodes(path, [fn], scope)
    draft_sync, draft_tokens = scope[fn.name](
        None, target_sync, batch.num_reqs, num_query_per_req=1
    )
    assert draft_tokens == target_sync.num_reqs
    assert draft_sync.uniform_token_count == 1
    assert draft_sync.num_tokens_across_dp.tolist() == [target_sync.num_reqs] * 4


@pytest.mark.parametrize("is_padding", [False, True])
def test_profile_padding_never_routes_extra_execution_rows(installed, is_padding):
    buffers = installed["InputBuffers"](8, 1024, torch.device("cpu"))
    batch = installed["InputBatch"].make_dummy(
        8, 1016, buffers, is_padding=is_padding, num_tokens_after_padding=1024
    )
    assert batch.num_tokens == 1016 and batch.num_tokens_after_padding == 1024
    assert batch.is_padding[:1016].eq(is_padding).all()
    assert batch.is_padding[1016:].all()
    assert batch.num_scheduled_tokens.sum() == 1016


@pytest.mark.parametrize(
    "eager,mode,expected",
    [
        (True, Mode.FULL_DECODE_ONLY, Mode.NONE),
        (False, Mode.FULL_DECODE_ONLY, Mode.FULL_DECODE_ONLY),
        (None, Mode.NONE, Mode.NONE),
    ],
)
def test_installed_draft_eager_mode_is_honored(eager, mode, expected):
    path = ROOT / "vllm/v1/worker/gpu/spec_decode/speculator.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "DraftModelSpeculator"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "resolve_cudagraph_mode"
    )
    scope = dict(CUDAGraphMode=Mode)
    compile_nodes(path, [fn], scope)
    assert (
        scope[fn.name](
            SimpleNamespace(speculative_config=SimpleNamespace(enforce_eager=eager)),
            mode,
        )
        == expected
    )
