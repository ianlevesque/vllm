# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU gates for observation-only startup diagnostics.

Exercise the actual helper and AST-loaded V2/worker control flow. Device work
is injected at its boundary; these tests do not qualify CUDA memory accounting,
asynchronous kernel attribution, complete model startup, or native sleep.
"""

import ast
import dataclasses
import gc
import json
import os
import sys
import weakref
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from test_shared_source_carries import ROOT, load

HELPER = "vllm/utils/startup_memory.py"
RUNNER = "vllm/v1/worker/gpu/model_runner.py"
WORKER = "vllm/v1/worker/gpu_worker.py"


@pytest.fixture
def diagnostic(monkeypatch):
    records, order = [], []

    def log(_format, payload):
        row = json.loads(payload)
        records.append(row)
        order.append((row["stage"], row["event"]))

    tree = ast.parse((ROOT / HELPER).read_text())
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.Import) and node.names[0].name == "vllm.envs")
        and not (isinstance(node, ast.ImportFrom) and node.module == "vllm.logger")
    ]
    scope = {
        "envs": NS(VLLM_STARTUP_MEMORY_DIAGNOSTICS=True),
        "init_logger": lambda _: NS(info=log),
    }
    exec(compile(tree, str(ROOT / HELPER), "exec"), scope)

    def proc():
        order.append(("read", "proc"))
        return {"MemAvailable_bytes": 99, "RssShmem_bytes": 23}

    def stats(_device):
        order.append(("read", "allocator"))
        return {"allocated_bytes.all.current": 17, "allocated_bytes.all.peak": 29}

    scope["_proc_memory"] = proc
    monkeypatch.setattr(torch.cuda, "memory_stats", stats)
    for module, names in (
        (
            torch.cuda,
            ("mem_get_info", "synchronize", "empty_cache", "reset_peak_memory_stats"),
        ),
        (
            torch.accelerator,
            (
                "get_memory_info",
                "synchronize",
                "empty_cache",
                "reset_peak_memory_stats",
            ),
        ),
    ):
        for name in names:
            monkeypatch.setattr(
                module,
                name,
                lambda *args, **kwargs: pytest.fail(
                    "diagnostic issued a device/allocator mutation call"
                ),
            )
    return NS(
        scope=scope,
        records=records,
        order=order,
        stage=scope["startup_memory_stage"],
        metadata=scope["startup_profile_metadata"],
    )


@pytest.mark.parametrize("flag, enabled", [(False, True), (True, False)])
def test_disabled_stage_does_not_read_or_log(diagnostic, flag, enabled):
    diagnostic.scope["envs"].VLLM_STARTUP_MEMORY_DIAGNOSTICS = flag
    metadata = lambda: pytest.fail("disabled diagnostics read metadata")
    with diagnostic.stage(
        "test", torch.device("cpu"), enabled=enabled, metadata=metadata
    ):
        pass
    assert diagnostic.order == []


@pytest.mark.parametrize("body_error", [None, RuntimeError, KeyboardInterrupt])
def test_entry_precedes_reads_and_exit_preserves_original_exception(
    diagnostic, body_error
):
    original = body_error("engine failure") if body_error else None

    def metadata():
        assert diagnostic.order == [("test", "begin")]
        return {"shape": [2, 4]}

    def run():
        with diagnostic.stage("test", torch.device("cpu"), metadata=metadata):
            diagnostic.order.append(("test", "body"))
            if body_error:
                raise original

    if body_error:
        with pytest.raises(body_error) as caught:
            run()
        assert caught.value is original
    else:
        run()
    ends = [r for r in diagnostic.records if r["event"] == "end"]
    assert len(ends) == 1 and ends[0]["status"] == ("error" if body_error else "ok")
    assert diagnostic.order.index(("test", "begin")) < diagnostic.order.index(
        ("read", "proc")
    )
    if body_error:
        assert diagnostic.order[-1] == ("test", "end")
    else:
        assert diagnostic.order.index(("test", "end")) < len(diagnostic.order) - 2
    assert len([x for x in diagnostic.order if x == ("read", "allocator")]) == (
        1 if body_error else 2
    )
    snapshot = next(
        r["values"] for r in diagnostic.records if r.get("reader") == "allocator"
    )
    assert snapshot["reserved_bytes.all.current"] is None


def test_failing_optional_readers_do_not_hide_engine_error(diagnostic, monkeypatch):
    def fail():
        assert diagnostic.records[0]["event"] == "begin"
        raise OSError("diagnostic unavailable")

    diagnostic.scope["_proc_memory"] = fail
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda _: fail())
    original = ValueError("engine")
    with (
        pytest.raises(ValueError) as caught,
        diagnostic.stage("test", torch.device("cpu"), metadata=fail),
    ):
        raise original
    assert caught.value is original
    assert len([r for r in diagnostic.records if r["event"] == "snapshot_error"]) == 2
    assert diagnostic.records[1]["event"] == "metadata_error"


def test_proc_fields_are_bytes_and_ignore_unrelated_values():
    texts = {
        "/proc/self/status": (
            "Name: worker\nVmRSS: 23 kB\nRssAnon: 7 kB\nRssShmem: 16 kB\nVmPin: 8 kB\n"
        ),
        "/proc/meminfo": "MemAvailable: 4096 kB\nShmem: 16 kB\nHugePages_Total: 0\n",
    }
    read = load(
        HELPER, "_proc_memory", Path=lambda path: NS(read_text=lambda: texts[path])
    )
    assert read() == {
        name + "_bytes": value * 1024
        for name, value in {
            "VmRSS": 23,
            "RssAnon": 7,
            "RssShmem": 16,
            "VmPin": 8,
            "MemAvailable": 4096,
            "Shmem": 16,
        }.items()
    }


def metadata_runner():
    return NS(
        model=NS(vision_tower=NS(config=NS(_attn_implementation="sdpa"))),
        max_num_tokens=8192,
        max_num_reqs=256,
        vllm_config=NS(
            model_config=NS(max_model_len=262144),
            attention_config=NS(backend="TRITON_ATTN"),
            compilation_config=NS(
                cudagraph_mode="FULL",
                static_forward_context={
                    "layer": NS(impl=NS()),
                },
            ),
        ),
        speculator=NS(
            num_speculative_steps=4,
            num_query_per_req=1,
            num_extra_query_per_req=0,
            hidden_size=1024,
            draft_max_seq_len=262144,
        ),
    )


@pytest.mark.parametrize("wrapper_depth", [0, 1, 2])
def test_known_vision_wrappers_and_absent_metadata(diagnostic, wrapper_depth):
    owner = metadata_runner()
    for _ in range(wrapper_depth):
        owner.model = NS(model=owner.model)
    metadata = diagnostic.metadata(owner)
    assert metadata["vision_path"] == "model." * wrapper_depth + "vision_tower"
    assert metadata["vision_attention_backend"] == "sdpa"
    absent = diagnostic.metadata(NS())
    assert absent["vision_path"] is None
    assert absent["vision_attention_backend"] is None
    assert absent["attention_backend"] is None
    assert absent["max_model_len"] is None
    assert absent["draft"]["hidden_size"] is None


def test_metadata_reads_real_mm_field_shapes_without_materializing_or_retaining(
    diagnostic,
):
    field = load(
        "vllm/multimodal/inputs.py",
        "MultiModalFieldElem",
        dataclass=dataclasses.dataclass,
    )
    # Meta tensors have no data to materialize: only shape/dtype/device are legal.
    tensor = torch.empty((2, 280, 768), dtype=torch.bfloat16, device="meta")
    ref = weakref.ref(tensor)
    owner = metadata_runner()
    with diagnostic.stage(
        "test",
        torch.device("cpu"),
        metadata=lambda tensor=tensor: diagnostic.metadata(
            owner, [("video", {"pixel_values": field(tensor, None)})]
        ),
    ):
        del tensor
        gc.collect()
        assert ref() is None
    data = next(
        row["values"] for row in diagnostic.records if row["event"] == "metadata"
    )
    assert data["input_items"] == 1 and data["input_modalities"] == ["video"]
    assert data["vision_attention_backend"] == "sdpa"
    assert data["draft"]["hidden_size"] == 1024
    assert data["draft"]["num_speculative_steps"] == 4
    assert data["max_num_reqs"] == 256 and data["max_model_len"] == 262144
    assert data["inputs"]["tensors"] == [
        {
            "path": "inputs[0][1].pixel_values.data",
            "shape": [2, 280, 768],
            "dtype": "torch.bfloat16",
            "device": "meta",
            "bytes": 860160,
        }
    ]
    summary = diagnostic.scope["_tensor_metadata"](
        [torch.empty(1, device="meta")] * 10000
    )
    assert summary["truncated"] and len(summary["tensors"]) < 128


def profile_runner(diagnostic, monkeypatch, fail_at=None):
    owner = metadata_runner()
    owner.device = torch.device("cpu")
    owner.max_num_tokens, owner.max_num_reqs = 8, 2
    owner.supports_mm_inputs = owner.is_first_pp_rank = owner.is_last_pp_rank = True
    owner.model_config = NS(multimodal_config=NS(skip_mm_profiling=False))
    owner.mm_registry = None
    owner.pooling_runner = owner.lora_config = None
    owner.model_memory_usage = 1
    owner.kv_connector = NS(set_disabled=lambda _: None)
    owner.maybe_dummy_run_with_lora = lambda *args, **kwargs: nullcontext()
    owner.step_timing = NS(
        forward_end=lambda: None, drafter_start=lambda: None, drafter_end=lambda: None
    )
    owner._draft_workspace_lane = None
    owner.req_states = NS(last_sampled_tokens=None, next_prefill_tokens=None)
    owner.sampler = NS(sampling_states=NS(temperature=NS(gpu=None), seeds=NS(gpu=None)))
    owner.speculator.supports_mm_inputs = False
    original = RuntimeError("failed " + str(fail_at))

    def operation(stage):
        diagnostic.order.append(("operation", stage))
        if stage == fail_at:
            raise original

    class Budget:
        def __init__(self, *args, **kwargs):
            operation("mm_inputs")

        def get_dummy_encoder_profile_inputs(self):
            return [("video", {"pixels": torch.empty(2, 3)})]

    def target(*args, **kwargs):
        operation("target")
        owner.execute_model_state = NS(
            input_batch=NS(num_reqs=2, num_tokens=8, logits_indices=[3, 7]),
            attn_metadata=None,
            slot_mappings_by_layer=None,
            hidden_states=torch.arange(32).reshape(8, 4),
            aux_hidden_states=None,
            dp_sync_state=None,
        )

    owner.execute_model = target
    owner.speculator.propose = lambda **kwargs: operation("draft")
    owner.model_state = NS(
        encoder_runner=NS(profile_encoder_cache=lambda *args: operation("encoder"))
    )
    owner._dummy_sampler_run = lambda states: operation("sampler")
    owner.reset_encoder_cache = lambda: operation("reset")
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: operation("cleanup"))
    common = dict(
        startup_memory_stage=diagnostic.stage,
        startup_profile_metadata=diagnostic.metadata,
    )
    dummy = load(
        RUNNER,
        "GPUModelRunner._dummy_run",
        **common,
        np=np,
        SchedulerOutput=NS(make_empty=lambda: NS()),
        get_drafter_hidden_states=lambda model, states: states,
        GPUWatermarkSampler=type("UnusedSampler", (), {}),
        use_workspace_lane=lambda _: nullcontext(),
    )
    owner._dummy_run = lambda *args, **kwargs: dummy(owner, *args, **kwargs)
    profile = load(
        RUNNER,
        "GPUModelRunner.profile_run",
        **common,
        MultiModalBudget=Budget,
        gc=NS(collect=lambda: operation("gc")),
    )
    owner.profile_run = lambda **kwargs: profile(owner, **kwargs)
    return owner, original


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "fail_at", [None, "mm_inputs", "encoder", "target", "draft", "sampler", "cleanup"]
)
def test_actual_v2_profile_bookends_and_work_order(
    diagnostic, monkeypatch, enabled, fail_at
):
    diagnostic.scope["envs"].VLLM_STARTUP_MEMORY_DIAGNOSTICS = enabled
    owner, original = profile_runner(diagnostic, monkeypatch, fail_at)
    if fail_at:
        with pytest.raises(RuntimeError) as caught:
            owner.profile_run()
        assert caught.value is original
    else:
        owner.profile_run()
    phases = ["mm_inputs", "encoder", "target", "draft", "sampler", "cleanup"]
    expected = phases[: phases.index(fail_at) + 1] if fail_at else phases
    operations = [name for kind, name in diagnostic.order if kind == "operation"]
    assert operations == expected + ([] if fail_at else ["reset", "gc"])
    begins = [r["stage"] for r in diagnostic.records if r["event"] == "begin"]
    ends = [r for r in diagnostic.records if r["event"] == "end"]
    if not enabled:
        assert not diagnostic.records
        assert not any(kind == "read" for kind, _ in diagnostic.order)
    else:
        assert begins == ["profile." + phase for phase in expected]
        assert [r["stage"] for r in ends] == begins
        assert [r["status"] for r in ends] == ["ok"] * (len(expected) - 1) + [
            "error" if fail_at else "ok"
        ]
        for phase in expected:
            assert diagnostic.order.index(
                ("profile." + phase, "begin")
            ) < diagnostic.order.index(("operation", phase))


def test_non_profile_dummy_run_does_not_emit_or_read(diagnostic, monkeypatch):
    owner, _ = profile_runner(diagnostic, monkeypatch)
    owner._dummy_run(8, is_profile=False)
    assert diagnostic.records == []
    assert diagnostic.order == [("operation", "target"), ("operation", "draft")]


def test_actual_worker_graph_phase_follows_regular_profile_and_preserves_failure(
    diagnostic, monkeypatch
):
    owner, _ = profile_runner(diagnostic, monkeypatch)
    original = RuntimeError("graph capture failed")

    def graphs():
        diagnostic.order.append(("operation", "graphs"))
        raise original

    @contextmanager
    def memory_profile(*args, **kwargs):
        yield NS()
        diagnostic.order.append(("operation", "memory_profile_end"))

    owner.profile_cudagraph_memory = graphs
    worker = NS(
        cache_config=NS(kv_cache_memory_bytes=None),
        model_runner=owner,
        init_snapshot=None,
        vllm_config=owner.vllm_config,
        device=owner.device,
        randomize_dummy_inputs=False,
        _scoped_allocator_max_split=lambda **kwargs: nullcontext(),
    )
    method = load(
        WORKER,
        "Worker.determine_available_memory",
        maybe_apply_startup_plan=lambda _: None,
        memory_profiling=memory_profile,
        set_current_vllm_config=lambda _: nullcontext(),
        current_platform=NS(is_cuda_alike=lambda: True),
        CUDAGraphMode=NS(NONE="NONE"),
        startup_memory_stage=diagnostic.stage,
        startup_profile_metadata=diagnostic.metadata,
    )
    with pytest.raises(RuntimeError) as caught:
        method(worker)
    assert caught.value is original
    assert diagnostic.order.index(
        ("operation", "memory_profile_end")
    ) < diagnostic.order.index(("profile.graphs", "begin"))
    end = [r for r in diagnostic.records if r["event"] == "end"][-1]
    assert end["stage"] == "profile.graphs" and end["status"] == "error"


def test_registered_switch_defaults_off_and_does_not_change_compile_key(monkeypatch):
    tree = ast.parse((ROOT / "vllm/envs.py").read_text())
    mapping = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and getattr(node.target, "id", None) == "environment_variables"
    )
    getter = next(
        value
        for key, value in zip(mapping.keys, mapping.values)
        if isinstance(key, ast.Constant)
        and key.value == "VLLM_STARTUP_MEMORY_DIAGNOSTICS"
    )
    read = eval(compile(ast.Expression(getter), "<registered env>", "eval"), {"os": os})
    monkeypatch.delenv("VLLM_STARTUP_MEMORY_DIAGNOSTICS", raising=False)
    assert read() is False
    monkeypatch.setenv("VLLM_STARTUP_MEMORY_DIAGNOSTICS", "1")
    assert read() is True
    monkeypatch.setitem(
        sys.modules, "vllm.config.utils", NS(normalize_value=lambda x: x)
    )
    factors = load(
        "vllm/envs.py",
        "compile_factors",
        environment_variables={
            "VLLM_STARTUP_MEMORY_DIAGNOSTICS": read,
            "SOME_COMPILE_OPTION": lambda: 7,
        },
    )
    enabled_factors = factors()
    monkeypatch.setenv("VLLM_STARTUP_MEMORY_DIAGNOSTICS", "0")
    assert factors() == enabled_factors
    assert "VLLM_STARTUP_MEMORY_DIAGNOSTICS" not in enabled_factors
    assert enabled_factors["SOME_COMPILE_OPTION"] == 7
