# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused CPU corruption controls for the native MiMo test guards."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from mimo_vision_sink_guards import GuardedEmptyLike, GuardedTensor, TensorSnapshot
from mimo_vision_sink_reference import (
    CASES,
    HEAD_DIM,
    assert_discriminating,
    dense_reference,
    make_inputs,
    relative_error,
)
from test_shared_source_carries import load


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_guarded_output_preserves_layout_and_requires_every_element_written(dtype):
    like = torch.zeros(3, 4, 64, dtype=dtype)
    guard = GuardedTensor(like, "output")
    assert guard.tensor.shape == like.shape and guard.tensor.stride() == like.stride()
    guard.check()
    with pytest.raises(AssertionError, match="unwritten sentinel"):
        guard.check(complete=True)
    guard.tensor.fill_(1)
    guard.check(complete=True)


@pytest.mark.parametrize("index,message", [(0, "prefix canary"), (-1, "suffix canary")])
def test_guard_detects_write_outside_payload(index, message):
    guard = GuardedTensor(torch.zeros(2, 4, 64), "output")
    guard.tensor.fill_(1)
    guard.storage[index] = 0
    with pytest.raises(AssertionError, match=message):
        guard.check(complete=True)


@pytest.mark.parametrize(
    "value,message",
    [
        (float("nan"), "nonfinite"),
        (float("inf"), "nonfinite"),
        (-97, "unwritten sentinel"),
    ],
)
def test_guard_rejects_partial_or_nonfinite_output(value, message):
    guard = GuardedTensor(torch.zeros(3, 4, 64), "output")
    guard.tensor.fill_(1)
    guard.tensor[1, 2, 3] = value
    with pytest.raises(AssertionError, match=message):
        guard.check(complete=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.int32])
def test_guarded_source_clone_detects_payload_mutation(dtype):
    original = torch.ones(7, dtype=dtype)
    guard = GuardedTensor(original, "source", source=True)
    guard.check()
    guard.tensor[3] = 2
    with pytest.raises(AssertionError, match="input bytes changed"):
        guard.check()
    assert torch.equal(original, torch.ones_like(original))


def test_snapshot_is_byte_exact_and_tracks_storage_rebinding():
    value = torch.tensor([0.0, 1.0])
    snapshot = TensorSnapshot(value, "source")
    value[0] = -0.0
    with pytest.raises(AssertionError, match="input bytes changed"):
        snapshot.check()
    value[0] = 0.0
    snapshot.check()
    value.data = value.clone()
    with pytest.raises(AssertionError, match="geometry changed"):
        snapshot.check()


def test_empty_like_adapter_only_substitutes_exact_caller_allocation(monkeypatch):
    like = torch.zeros(3, 4, 64, dtype=torch.bfloat16)
    adapter = GuardedEmptyLike(like)
    monkeypatch.setattr(torch, "empty_like", adapter)
    output = torch.empty_like(like)
    assert output is adapter.output.tensor and adapter.calls == 1
    other = torch.empty_like(torch.zeros(2))
    assert other.shape == (2,) and adapter.calls == 1
    output.fill_(1)
    adapter.output.check(complete=True)
    with pytest.raises(AssertionError, match="more than once"):
        torch.empty_like(like)


@pytest.mark.parametrize(
    "fault,message",
    [
        (None, None),
        ("input", "input bytes"),
        ("prefix", "prefix canary"),
        ("unwritten", "unwritten sentinel"),
        ("nonfinite", "nonfinite"),
        ("metadata", "input bytes"),
    ],
)
def test_actual_native_runner_detects_boundary_corruption(monkeypatch, fault, message):
    """Execute unchanged runner/caller bodies; only CUDA execution is injected."""
    kernel_name = "vllm.v1.attention.ops.triton_prefill_attention"
    kernel = ModuleType(kernel_name)
    calls, synchronizations = [], []

    def cpu_kernel(q, k, v, output, **options):
        calls.append(options)
        starts, lengths = options["b_start_loc"], options["b_seq_len"]
        cu = torch.cat([starts, (starts[-1] + lengths[-1]).reshape(1)])
        sinks = options["sinks"]
        mode = (
            "none"
            if sinks is None
            else ("key0" if options["sinks_bias_key0"] else "denominator")
        )
        reference = dense_reference(
            q.clone(),  # CPU oracle allocates scratch; the real Triton kernel does not.
            k,
            v,
            cu,
            sinks,
            options["softmax_scale"],
            options["sliding_window_q"],
            mode,
        )
        output.copy_(reference)
        if fault == "input":
            q[0, 0, 0] += 1
        elif fault == "prefix":
            output.as_strided((1,), (1,), storage_offset=0).fill_(0)
        elif fault == "unwritten":
            output[0, 0, 0] = -97
        elif fault == "nonfinite":
            output[0, 0, 0] = float("nan")
        elif fault == "metadata":
            lengths[0] += 1

    kernel.context_attention_fwd = cpu_kernel
    package = ModuleType("vllm.v1.attention.ops")
    package.triton_prefill_attention = kernel
    monkeypatch.setitem(sys.modules, kernel_name, kernel)
    monkeypatch.setitem(sys.modules, package.__name__, package)
    model = ModuleType("vllm.model_executor.models.mimo_v2_omni")
    model.MiMoVisionAttention = SimpleNamespace(
        _forward_window_attn=load(
            "vllm/model_executor/models/mimo_v2_omni.py",
            "MiMoVisionAttention._forward_window_attn",
        )
    )
    monkeypatch.setitem(sys.modules, model.__name__, model)
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda: synchronizations.append(True)
    )
    run = load(
        "tests/models/multimodal/test_mimo_v2_omni.py",
        "run_window_case",
        SimpleNamespace=SimpleNamespace,
        GuardedTensor=GuardedTensor,
        GuardedEmptyLike=GuardedEmptyLike,
        TensorSnapshot=TensorSnapshot,
        HEAD_DIM=HEAD_DIM,
        make_inputs=lambda case, dtype, device: make_inputs(case, dtype),
        dense_reference=dense_reference,
        assert_discriminating=assert_discriminating,
        relative_error=relative_error,
    )
    if fault is None:
        result = run(monkeypatch, torch.bfloat16, CASES[-1])
        assert result["attention_invocations"] == len(calls) == 3
        assert result["input_bytes_and_canaries_unchanged"]
        assert result["outputs_finite_fully_written"]
        assert result["device_synchronized_closure"]
    else:
        with pytest.raises(AssertionError, match=message):
            run(monkeypatch, torch.bfloat16, CASES[-1])
        assert len(calls) == 1  # No later negative control after corruption.
    assert synchronizations
