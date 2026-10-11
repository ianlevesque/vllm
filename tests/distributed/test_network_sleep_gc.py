# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CPython GC / actual Worker methods, isolated from pytest's own heap."""

import ast
import gc
import json
import os
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(os.environ.get("VLLM_GC_SOURCE", Path(__file__).parents[2]))


def _functions(path, names, class_name=None):
    body = ast.parse(path.read_text()).body
    if class_name:
        body = next(
            n.body for n in body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
    selected = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    return compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec")


def _probe(frozen, enabled, failure):
    # Avoid automatic collection between explicit unfreeze and the zero-policy
    # control. gc.isenabled() still exercises both states. Manual collections
    # in the *real* upstream capture/heap helpers continue to run.
    gc.set_threshold(10**9, 10**9, 10**9)
    scope = {
        "gc": gc,
        "contextmanager": contextmanager,
        "envs": SimpleNamespace(VLLM_ENABLE_CUDAGRAPH_GC=False),
    }
    exec(
        _functions(
            ROOT / "vllm/utils/gc_utils.py",
            {"freeze_gc_heap", "freeze_gc_for_cudagraph_capture"},
        ),
        scope,
    )
    freeze = scope["freeze_gc_heap"]
    capture_gc = scope["freeze_gc_for_cudagraph_capture"]
    freezing, events = [], []
    waking = False

    def freeze_heap():
        if waking:
            assert worker._network_sleep_groups == {}
            assert events[-1] == "synchronized"
        freeze()
        freezing.append(gc.get_freeze_count())

    def capture():
        with capture_gc():
            if failure == "capture":
                raise RuntimeError("capture failure")
        events.append("captured")

    def initialize(*args, **kwargs):
        if failure == "initialize":
            raise RuntimeError("initialize failure")

    def synchronize():
        if waking and failure == "synchronize":
            raise RuntimeError("synchronize failure")
        events.append("synchronized")

    def clear_graphs(worker):
        gc.unfreeze()
        gc.collect()

    module = ModuleType("vllm.distributed.network_sleep")
    module.check_network_sleep = lambda worker: None
    module.worker_global_rank = lambda worker: 0
    module.live_groups = lambda: {"tp:0": object()}
    module.clear_collective_graphs = clear_graphs
    module.close_transport = lambda groups: None
    module.restore_group_references = lambda previous: None
    sys.modules[module.__name__] = module
    worker_scope = {
        "Any": object,
        "gc": gc,
        "os": os,
        "set_current_vllm_config": lambda _: nullcontext(),
        "init_worker_distributed_environment": initialize,
        "current_platform": SimpleNamespace(dist_backend="nccl"),
        "logger": SimpleNamespace(info=lambda *a: None),
        "freeze_gc_heap": freeze_heap,
    }
    names = {
        "network_sleep_status",
        "sleep_network",
        "wake_network",
        "_freeze_gc_heap",
    }
    exec(
        _functions(ROOT / "vllm/v1/worker/gpu_worker.py", names, "Worker"),
        worker_scope,
    )
    worker_class = type(
        "ActualWorkerMethods",
        (),
        {name: worker_scope[name] for name in names if name in worker_scope},
    )
    worker = worker_class()
    worker.rank = worker.local_rank = 0
    worker.distributed_init_method = "tcp://not-contacted:1"
    worker.vllm_config = object()
    worker.model_runner = SimpleNamespace(capture_model=capture)
    worker.synchronize_device = synchronize
    # These remain strongly reachable throughout the probe. Counting never
    # enumerates the heap, and 1000 is only a test sentinel, not runtime policy.
    sentinels = [[n] for n in range(1000)]
    gc.unfreeze()
    (gc.enable if enabled else gc.disable)()
    if frozen:
        if hasattr(worker, "_freeze_gc_heap"):
            worker._freeze_gc_heap()
        else:
            # Negative control on the original source: startup called this
            # upstream helper directly before the explicit policy was added.
            freeze_heap()
    startup_count = gc.get_freeze_count()
    assert not frozen or startup_count >= len(sentinels)
    counts = []
    for generation in (1, 2):
        waking = False
        worker.sleep_network()
        waking = True
        before = len(freezing)
        if failure:
            with pytest.raises(RuntimeError, match=failure):
                worker.wake_network()
            assert len(freezing) == before
            assert worker._network_sleep_state == "restoring"
            assert gc.get_freeze_count() < len(sentinels)
            assert gc.isenabled() == enabled
            break
        status = worker.wake_network()
        assert status["generation"] == generation and status["state"] == "active"
        assert gc.isenabled() == enabled
        count = gc.get_freeze_count()
        assert (count >= len(sentinels)) == frozen, (startup_count, count, frozen)
        assert len(freezing) == before + int(frozen)
        assert worker.wake_network() == status
        assert len(freezing) == before + int(frozen), "active wake must be idempotent"
        counts.append(count)
    print(json.dumps({"startup_count": startup_count, "wake_counts": counts}))
    gc.unfreeze()
    gc.enable()


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("failure", ["", "initialize", "capture", "synchronize"])
def test_network_wake_preserves_startup_gc_policy(frozen, enabled, failure):
    result = subprocess.run(
        [sys.executable, __file__, str(int(frozen)), str(int(enabled)), failure],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _probe(bool(int(sys.argv[1])), bool(int(sys.argv[2])), sys.argv[3])
