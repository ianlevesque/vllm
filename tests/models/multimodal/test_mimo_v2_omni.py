# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native MiMo sink arithmetic through the actual caller (local TP shapes).

CPU routing and full-attention branch tests live in tests/standalone.
TP8 local shapes do not exercise NCCL or an eight-rank model.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.standalone.mimo_vision_sink_guards import (
    GuardedEmptyLike,
    GuardedTensor,
    TensorSnapshot,
)
from tests.standalone.mimo_vision_sink_reference import (
    CASES,
    HEAD_DIM,
    assert_discriminating,
    dense_reference,
    make_inputs,
    relative_error,
)
from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires Triton CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_window_attention_applies_denominator_sink(monkeypatch, dtype, case):
    run_window_case(monkeypatch, dtype, case)


def run_window_case(monkeypatch, dtype, case):
    from vllm.model_executor.models.mimo_v2_omni import MiMoVisionAttention
    from vllm.v1.attention.ops import triton_prefill_attention as kernel

    q, k, v, cu, global_sinks, sinks, window = make_inputs(case, dtype, "cuda")
    _, heads, _, tp, rank, _, lengths = case
    source_guards = [
        GuardedTensor(tensor, name, source=True)
        for name, tensor in zip(
            ("q", "k", "v", "cu", "global_sinks"), (q, k, v, cu, global_sinks)
        )
    ]
    q, k, v, cu, global_sinks = [guard.tensor for guard in source_guards]
    head_start = rank * (heads // tp)
    sinks = global_sinks[head_start : head_start + heads // tp]
    q_ref, k_ref, v_ref, cu_ref, global_ref = [
        guard.snapshot.value for guard in source_guards
    ]
    sinks_ref = global_ref[head_start : head_start + heads // tp]
    allocation = GuardedEmptyLike(q)
    output_guards = [allocation.output]
    completed_outputs = set()
    completed_snapshots = []
    attention_calls = 0
    owner = SimpleNamespace(
        tp_rank=rank,
        num_heads_per_partition=heads // tp,
        sinks=global_sinks,
        scale=HEAD_DIM**-0.5,
        visual_token_window_size=window,
    )
    calls = []
    original = kernel.context_attention_fwd

    def closure():
        # No CPU inspection of asynchronous writes before device completion.
        torch.cuda.synchronize()
        for guard in source_guards:
            guard.check()
        for guard in output_guards:
            guard.check(complete=id(guard.tensor) in completed_outputs)
        for snapshot in completed_snapshots:
            snapshot.check()

    def audited_kernel(*args, **kwargs):
        nonlocal attention_calls
        metadata = [
            TensorSnapshot(kwargs[name], name)
            for name in ("b_start_loc", "b_seq_len", "sinks")
            if kwargs[name] is not None
        ]
        attention_calls += 1
        try:
            original(*args, **kwargs)
            torch.cuda.synchronize()
            completed_outputs.add(id(args[3]))
        finally:
            closure()
            for snapshot in metadata:
                snapshot.check()
        completed_snapshots.append(TensorSnapshot(args[3], "completed-output"))

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return audited_kernel(*args, **kwargs)

    monkeypatch.setattr(kernel, "context_attention_fwd", capture)
    # Only the caller's actual empty_like(q) receives a guarded contiguous view.
    # The pinned Triton function has no separately allocated external scratch.
    with monkeypatch.context() as allocation_patch:
        allocation_patch.setattr(torch, "empty_like", allocation)
        output = MiMoVisionAttention._forward_window_attn(
            owner, q, k, v, cu, max(lengths)
        )
    assert allocation.calls == 1 and output is allocation.output.tensor
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is q and args[1] is k and args[2] is v and args[3] is output
    assert kwargs["sinks_bias_key0"] is False
    assert kwargs["is_causal"] is False
    assert kwargs["sliding_window_q"] == kwargs["sliding_window_k"] == window
    assert kwargs["softmax_scale"] == owner.scale
    assert kwargs["max_input_len"] == max(lengths)
    torch.testing.assert_close(kwargs["sinks"], sinks, rtol=0, atol=0)
    torch.testing.assert_close(kwargs["b_start_loc"], cu[:-1], rtol=0, atol=0)
    torch.testing.assert_close(kwargs["b_seq_len"], cu[1:] - cu[:-1], rtol=0, atol=0)
    reference = dense_reference(
        q_ref, k_ref, v_ref, cu_ref, sinks_ref, owner.scale, window
    )
    assert torch.isfinite(reference).all().item()
    controls = assert_discriminating(
        q_ref, k_ref, v_ref, cu_ref, sinks_ref, owner.scale, window, reference
    )
    error = relative_error(output, reference)
    tolerance = 1e-2 if dtype == torch.bfloat16 else 3e-3
    assert error < tolerance, (case[0], str(dtype), error, controls)
    sequence_head_errors = []
    for start, end in zip(cu[:-1].tolist(), cu[1:].tolist()):
        diff = output[start:end].float() - reference[start:end]
        head_errors = (
            diff.square().sum((0, 2)) / reference[start:end].square().sum((0, 2))
        ).sqrt()
        sequence_head_errors.extend(head_errors.tolist())
    assert max(sequence_head_errors) < tolerance, sequence_head_errors
    # Execute the two incorrect kernel modes as negative controls. They must
    # match their own dense references AND disagree with the null-logit oracle.
    native_controls = {}
    for mode in ("key0", "none"):
        wrong_guard = GuardedTensor(q, mode + "-output")
        output_guards.append(wrong_guard)
        wrong = wrong_guard.tensor
        options = dict(kwargs, sinks_bias_key0=mode == "key0")
        if mode == "none":
            options["sinks"] = None
        audited_kernel(q, k, v, wrong, **options)
        wrong_reference = dense_reference(
            q_ref, k_ref, v_ref, cu_ref, sinks_ref, owner.scale, window, mode
        )
        assert torch.isfinite(wrong_reference).all().item()
        assert relative_error(wrong, wrong_reference) < tolerance
        native_controls[mode] = relative_error(wrong, reference)
        assert native_controls[mode] > 0.05
        assert relative_error(wrong[-1:], reference[-1:]) > 0.05
    closure()
    assert attention_calls == 3 and len(completed_outputs) == 3
    return {
        "attention_invocations": attention_calls,
        "input_guard_buffers": len(source_guards),
        "output_guard_buffers": len(output_guards),
        "input_bytes_and_canaries_unchanged": True,
        "outputs_finite_fully_written": True,
        "device_synchronized_closure": True,
        "case": case[0],
        "dtype": str(dtype),
        "error": error,
        "max_sequence_head_error": max(sequence_head_errors),
        "dense_controls": controls,
        "native_controls": native_controls,
        "actual_caller_sinks_bias_key0": kwargs["sinks_bias_key0"],
    }
