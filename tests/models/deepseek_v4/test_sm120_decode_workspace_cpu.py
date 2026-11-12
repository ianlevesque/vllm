# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sizing of the FlashInfer SM120 sparse-MLA decode workspace, without a GPU."""

import pytest
import torch

import vllm.envs as envs
from vllm.models.deepseek_v4.nvidia import flashinfer_sparse as fs

MIB = 1024 * 1024
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def small_workspace(monkeypatch):
    monkeypatch.setattr(fs, "_FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE", MIB)
    monkeypatch.setattr(fs, "_flashinfer_dsv4_workspace_by_device", {})
    monkeypatch.setattr(fs, "_flashinfer_dsv4_workspace_captured", set())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.delenv("VLLM_DSV4_FLASHINFER_WORKSPACE_MB", raising=False)


def test_scratch_formula_matches_flashinfer_resolve_execution():
    # slots = rows * scratch_heads * (ceil(topk/64) + ceil(extra/64));
    # BF16 partials of 512 values + FP32 split LSE per slot, then a final LSE.
    assert fs.sm120_sparse_decode_workspace_bytes(1, 16, 128, 0) == 16 * 2 * 1028 + 64 + 48
    # 8 heads keep an 8-head scratch; anything else rounds up to 16.
    assert fs.sm120_sparse_decode_workspace_bytes(1, 8, 64, 0) == 8 * 1028 + 32 + 48
    assert fs.sm120_sparse_decode_workspace_bytes(1, 24, 64, 0) == 32 * 1028 + 128 + 48
    # Partial tiles count as whole chunks on both tables.
    assert fs.sm120_sparse_decode_workspace_bytes(1, 16, 129, 1) == 16 * 4 * 1028 + 64 + 48
    # The DeepSeek V4 Flash shapes that exhausted the fixed 128 MiB buffer: a
    # 1M-token compressed layer (8192 candidates) at the kernel's 64-row ceiling.
    tp4 = fs.sm120_sparse_decode_workspace_bytes(64, 32, 128, 8192)
    assert tp4 == 64 * 32 * 130 * 1028 + 64 * 32 * 4 + 48
    assert 261 * MIB < tp4 < 262 * MIB
    tp8 = fs.sm120_sparse_decode_workspace_bytes(64, 16, 128, 8192)
    assert 130 * MIB < tp8 < 131 * MIB  # still above the old fixed buffer
    assert fs.sm120_sparse_decode_workspace_bytes(64, 8, 128, 8192) < 128 * MIB  # TP16 fits


def test_small_launches_keep_the_default_buffer():
    workspace = fs.get_flashinfer_dsv4_decode_workspace(CPU, rows=1, heads=8, topk=64, extra_topk=0)
    assert workspace.numel() == MIB
    assert fs.get_flashinfer_dsv4_workspace(CPU) is workspace


def test_first_eager_launch_grows_for_the_largest_captured_batch():
    need = fs.sm120_sparse_decode_workspace_bytes(64, 16, 128, 1024)
    workspace = fs.get_flashinfer_dsv4_decode_workspace(
        CPU, rows=2, heads=16, topk=128, extra_topk=1024, capture_rows=64
    )
    assert need <= workspace.numel() < need * 1.1 + MIB
    assert workspace.numel() % MIB == 0
    # Every later launch, eager or captured, sees the same tensor.
    again = fs.get_flashinfer_dsv4_decode_workspace(
        CPU, rows=64, heads=16, topk=128, extra_topk=1024, capture_rows=64
    )
    assert again is workspace and fs.get_flashinfer_dsv4_workspace(CPU) is workspace


def test_rows_are_capped_at_the_split_decode_kernel_ceiling(monkeypatch):
    # 512 captured rows never reach the split-K kernel: size for 64, not 512.
    at_ceiling = fs.sm120_sparse_decode_workspace_bytes(64, 16, 128, 1024)
    workspace = fs.get_flashinfer_dsv4_decode_workspace(
        CPU, rows=512, heads=16, topk=128, extra_topk=1024, capture_rows=512
    )
    assert at_ceiling <= workspace.numel() < at_ceiling * 1.1 + MIB
    # The ceiling is FlashInfer's own constant when it is importable.
    monkeypatch.setattr(fs, "_sm120_split_decode_max_rows", lambda: 8)
    monkeypatch.setattr(fs, "_flashinfer_dsv4_workspace_by_device", {})
    small = fs.get_flashinfer_dsv4_decode_workspace(
        CPU, rows=512, heads=16, topk=128, extra_topk=1024, capture_rows=512
    )
    assert small.numel() < workspace.numel()


def test_capture_never_reallocates_and_names_the_knob(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="VLLM_DSV4_FLASHINFER_WORKSPACE_MB"):
        fs.get_flashinfer_dsv4_decode_workspace(CPU, rows=64, heads=16, topk=128, extra_topk=1024)
    assert fs.get_flashinfer_dsv4_workspace(CPU).numel() == MIB


def test_growth_is_refused_once_a_graph_replays_the_buffer(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    captured = fs.get_flashinfer_dsv4_decode_workspace(CPU, rows=1, heads=8, topk=64, extra_topk=0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with pytest.raises(RuntimeError, match="too small for a captured decode launch"):
        fs.get_flashinfer_dsv4_decode_workspace(CPU, rows=64, heads=16, topk=128, extra_topk=1024)
    assert fs.get_flashinfer_dsv4_workspace(CPU) is captured


def test_fixed_size_is_honoured_and_never_grown(monkeypatch):
    monkeypatch.setenv("VLLM_DSV4_FLASHINFER_WORKSPACE_MB", "3")
    assert envs.VLLM_DSV4_FLASHINFER_WORKSPACE_MB == 3
    workspace = fs.get_flashinfer_dsv4_workspace(CPU)
    assert workspace.numel() == 3 * MIB
    # Eager launches beyond it fall back to FlashInfer's own temporary scratch.
    same = fs.get_flashinfer_dsv4_decode_workspace(CPU, rows=64, heads=16, topk=128, extra_topk=1024)
    assert same is workspace
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="VLLM_DSV4_FLASHINFER_WORKSPACE_MB to at least"):
        fs.get_flashinfer_dsv4_decode_workspace(CPU, rows=64, heads=16, topk=128, extra_topk=1024)
