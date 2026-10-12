# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU policy coverage using the actual worker context manager's AST.

Native allocation/reuse coverage lives in tests/v1/worker/test_gpu_worker.py.
"""

import ast
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def worker_scope(*, expandable, cuda_alike=True, backend="native", getter=True):
    source = Path(__file__).parents[2] / "vllm/v1/worker/gpu_worker.py"
    worker = next(
        x
        for x in ast.parse(source.read_text()).body
        if isinstance(x, ast.ClassDef) and x.name == "Worker"
    )
    method = next(
        x
        for x in worker.body
        if isinstance(x, ast.FunctionDef) and x.name == "_scoped_allocator_max_split"
    )
    original = (
        "garbage_collection_threshold:0.8,roundup_power2_divisions:[256:1,512:2,>:4], "
    )
    set_settings = Mock()
    get_settings = Mock(return_value=original)
    snapshot = Mock(
        return_value={"allocator_settings": {"PYTORCH_CUDA_ALLOC_CONF": original}}
    )
    c = SimpleNamespace(_accelerator_setAllocatorSettings=set_settings)
    if getter:
        c._accelerator_getAllocatorSettings = get_settings
    live = Mock(return_value=expandable)
    namespace = {
        "contextmanager": contextmanager,
        "current_platform": SimpleNamespace(is_cuda_alike=lambda: cuda_alike),
        "expandable_segments_enabled": live,
        "torch": SimpleNamespace(
            _C=c,
            cuda=SimpleNamespace(
                memory=SimpleNamespace(
                    get_allocator_backend=lambda: backend, _snapshot=snapshot
                )
            ),
        ),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace[method.name], original, live, get_settings, set_settings, snapshot


@pytest.mark.parametrize("expandable", [True, False, None])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("getter", [False, True])
def test_live_state_controls_scope_and_exact_restoration(
    monkeypatch, expandable, fail, getter
):
    # The environment intentionally disagrees with the live setting.
    monkeypatch.setenv(
        "PYTORCH_CUDA_ALLOC_CONF", f"expandable_segments:{not expandable}"
    )
    scope, original, live, get, set_, snapshot = worker_scope(
        expandable=expandable, getter=getter
    )
    expected = (
        pytest.raises(RuntimeError, match="profiling failed") if fail else nullcontext()
    )
    with expected, scope(None, 20):
        if expandable is False:
            set_.assert_called_once_with(
                original.strip().rstrip(",") + ",max_split_size_mb:20"
            )
        else:
            set_.assert_not_called()
        if fail:
            raise RuntimeError("profiling failed")
    live.assert_called_once_with()
    if expandable is False:
        assert set_.call_count == 2
        set_.assert_called_with(original)
        assert get.call_count == int(getter)
        assert snapshot.call_count == int(not getter)
    else:
        get.assert_not_called()
        set_.assert_not_called()
        snapshot.assert_not_called()


@pytest.mark.parametrize(
    "cuda_alike,backend", [(False, "native"), (True, "cudaMallocAsync")]
)
def test_other_allocators_are_untouched(cuda_alike, backend):
    scope, _, live, get, set_, snapshot = worker_scope(
        expandable=False, cuda_alike=cuda_alike, backend=backend
    )
    with scope(None, 20):
        pass
    for mock in (live, get, set_, snapshot):
        mock.assert_not_called()
