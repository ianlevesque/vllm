# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the SM12x RoCE fused-collective dispatch.

The kernels need GB10 hardware and a RoCE fabric (they are validated there
bit for bit against rank-ordered sums); these tests cover what decides, on
every rank identically, which collective a tensor takes: the shape and size
eligibility and the adapter's routing order.
"""

import pytest
import torch

from vllm.distributed.device_communicators.b12x_roce_all_reduce import (
    B12xRoceAllReduce,
)
from vllm.models.kimi_k3.nvidia.ops.sm121.roce_fused import (
    gather_geometry_ok,
    moe_tail_geometry_ok,
    two_shot_geometry_ok,
)

WORLD = 16
SLOT = 2 * 1024 * 1024


@pytest.mark.parametrize(
    "rows,columns,expected",
    [
        (1, 7168, True),
        (6, 7168, True),
        (6, 3584, True),
        (146, 7168, True),  # 16 windows of 146 * 896 bytes just fit a 2 MB slot
        (147, 7168, False),
        (6, 7160, False),  # not whole 16-byte packs per rank
        (0, 7168, False),
        (6, 0, False),
    ],
)
def test_two_shot_geometry(rows, columns, expected):
    assert two_shot_geometry_ok(rows, columns, WORLD, SLOT) is expected


@pytest.mark.parametrize(
    "rows,latent,columns,expected",
    [
        (1, 3584, 7168, True),
        (6, 3584, 7168, True),
        (9, 3584, 7168, True),  # 9 rows * 14 warps = 126 of the cluster's 128
        (10, 3584, 7168, False),
        (6, 3520, 7168, False),  # a latent row must be whole warps of packs
        (6, 3584, 7160, False),
        (4, 1024, 1024, True),
    ],
)
def test_moe_tail_geometry(rows, latent, columns, expected):
    assert moe_tail_geometry_ok(rows, latent, columns, WORLD, SLOT) is expected


@pytest.mark.parametrize(
    "row_bytes,rows,expected",
    [
        ((896,), 6, True),
        ((224, 448), 6, True),  # FP32 router logits + BF16 latent shard
        ((224, 448), 4100, False),  # 2.75 MB: two rounds instead
        ((224,), 4100, True),
        ((450,), 6, False),  # rows must be whole 16-byte packs
        ((896, 896, 896), 6, False),
        ((896,), 0, False),
    ],
)
def test_gather_geometry(row_bytes, rows, expected):
    assert gather_geometry_ok(row_bytes, rows, SLOT) is expected


class _FakeFused:
    def __init__(self):
        self.calls = []

    def can_two_shot_all_reduce(self, inp):
        return inp.dtype == torch.bfloat16 and inp.ndim == 2 and inp.is_contiguous()

    def two_shot_all_reduce(self, inp):
        self.calls.append(("two_shot", tuple(inp.shape)))
        return inp * WORLD

    def can_all_gather(self, *tensors):
        return all(t.ndim == 2 and t.is_contiguous() for t in tensors)

    def all_gather(self, first, second=None):
        self.calls.append(("gather", tuple(first.shape)))
        return torch.cat([first] * WORLD, dim=-1)


class _FakeRuntime:
    def __init__(self):
        self.calls = []

    def should_allreduce(self, inp):
        return inp.is_contiguous() and inp.numel() * inp.element_size() <= SLOT

    def all_reduce(self, inp):
        self.calls.append(("flat", tuple(inp.shape), inp.dtype))
        return inp * WORLD

    def all_gather(self, inp, dim):
        self.calls.append(("legacy_gather", tuple(inp.shape), dim))
        return torch.cat([inp] * WORLD, dim=dim)


def _adapter(fused, hierarchy=True, minimum=24 * 1024):
    adapter = object.__new__(B12xRoceAllReduce)
    adapter.gather_only = False
    adapter.disabled = False
    adapter.rank = 1  # a non-zero rank logs at debug level
    adapter.world_size = WORLD
    adapter._runtime = _FakeRuntime()
    adapter._hierarchical_runtimes = (
        [_FakeRuntime(), _FakeRuntime()] if hierarchy else []
    )
    adapter._hierarchical_min_size = 64 * 1024
    adapter._hierarchical_max_size = SLOT // 2
    adapter._announced = adapter._announced_gather = True
    adapter._announced_hierarchy = adapter._announced_two_shot = True
    adapter.fused = fused
    adapter._two_shot_min_size = minimum
    return adapter


def test_two_shot_supersedes_the_hierarchy_above_the_minimum():
    fused = _FakeFused()
    adapter = _adapter(fused)
    hidden = torch.zeros(6, 7168, dtype=torch.bfloat16)  # 86 KB
    adapter.custom_all_reduce(hidden)
    assert fused.calls == [("two_shot", (6, 7168))]
    assert not adapter._runtime.calls
    assert not adapter._hierarchical_runtimes[0].calls


def test_small_payloads_stay_on_the_flat_all_reduce():
    fused = _FakeFused()
    adapter = _adapter(fused)
    adapter.custom_all_reduce(torch.zeros(1, 7168, dtype=torch.bfloat16))  # 14 KB
    assert not fused.calls
    assert adapter._runtime.calls == [("flat", (1, 7168), torch.bfloat16)]


def test_other_shapes_and_dtypes_keep_their_path():
    fused = _FakeFused()
    adapter = _adapter(fused)
    adapter.custom_all_reduce(torch.zeros(6, 7168, dtype=torch.float32))
    adapter.custom_all_reduce(torch.zeros(2, 6, 7168, dtype=torch.bfloat16))
    assert not fused.calls
    assert [c[0] for c in adapter._runtime.calls] == ["flat"]
    # The 3-D BF16 payload is in the hierarchy's range.
    assert [c[0] for c in adapter._hierarchical_runtimes[0].calls] == ["flat"]


def test_without_fused_collectives_nothing_changes():
    adapter = _adapter(None)
    adapter.custom_all_reduce(torch.zeros(6, 7168, dtype=torch.bfloat16))
    assert [c[0] for c in adapter._hierarchical_runtimes[0].calls] == ["flat"]
    assert adapter.all_gather(torch.zeros(6, 448), dim=-1).shape == (6, 448 * WORLD)
    assert adapter._runtime.calls == [("legacy_gather", (6, 448), -1)]


def test_last_dim_gathers_use_the_cluster_kernel():
    fused = _FakeFused()
    adapter = _adapter(fused)
    assert adapter.all_gather(torch.zeros(6, 448), dim=-1).shape == (6, 448 * WORLD)
    assert adapter.all_gather(torch.zeros(6, 448), dim=1).shape == (6, 448 * WORLD)
    assert fused.calls == [("gather", (6, 448)), ("gather", (6, 448))]
    adapter.all_gather(torch.zeros(6, 448), dim=0)
    assert adapter._runtime.calls == [("legacy_gather", (6, 448), 0)]


GB10_SHARED = 101376  # opt-in shared memory per block on SM12.1


@pytest.mark.parametrize(
    "rows,k,columns,shared,expected",
    [
        (1, 768, 7168, GB10_SHARED, True),  # attention output projection
        (6, 768, 7168, GB10_SHARED, True),
        (16, 768, 7168, GB10_SHARED, True),  # one MMA covers sixteen rows
        (17, 768, 7168, GB10_SHARED, False),
        (6, 3584, 448, GB10_SHARED, True),  # the latent-MoE up-projection shard
        (13, 3584, 448, GB10_SHARED, True),  # 91 KB of x + 8 KB of accumulators
        (14, 3584, 448, GB10_SHARED, False),  # x exceeds a GB10 block
        (14, 3584, 448, 232448, True),
        (6, 776, 7168, GB10_SHARED, False),  # K must be whole 32-feature steps
        (6, 768, 7164, GB10_SHARED, False),  # output columns must be whole packs
        (0, 768, 7168, GB10_SHARED, False),
    ],
)
def test_gemm_geometry(rows, k, columns, shared, expected):
    from vllm.models.kimi_k3.nvidia.ops.sm121.roce_fused import gemm_geometry_ok

    assert gemm_geometry_ok(rows, k, columns, shared) is expected


class _FakeLinear:
    def __init__(self, weight, bias=None, quantized=False):
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        self.weight = weight
        self.bias = bias
        self.quant_method = object() if quantized else UnquantizedLinearMethod()
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return x.float() @ self.weight.float().t(), None


class _FakeGemmFused:
    def __init__(self, eligible=True):
        self.eligible = eligible
        self.launched = []

    def can_gemm_all_reduce(self, x, weight):
        return self.eligible

    def gemm_all_reduce(self, x, weight):
        self.launched.append((tuple(x.shape), tuple(weight.shape)))
        return torch.zeros(x.shape[0], weight.shape[0], dtype=x.dtype)


def test_sm121_gemm_ar_accepts_only_unquantized_bf16_projections():
    from vllm.models.kimi_k3.nvidia.ops.sm121.gemm_ar import Sm121GemmAr

    backend = Sm121GemmAr(7168)
    weight = torch.zeros(7168, 768, dtype=torch.bfloat16)
    assert backend.can_run(_FakeLinear(weight))
    assert not backend.can_run(_FakeLinear(weight, quantized=True))
    assert not backend.can_run(_FakeLinear(weight, bias=torch.zeros(7168)))
    assert not backend.can_run(_FakeLinear(weight.float()))
    assert not backend.can_run(
        _FakeLinear(torch.zeros(3584, 768, dtype=torch.bfloat16))
    )
    assert not backend.can_run(
        _FakeLinear(torch.zeros(7168, 776, dtype=torch.bfloat16))
    )


def test_sm121_gemm_ar_runs_decode_batches_only(monkeypatch):
    from vllm.models.kimi_k3.nvidia.ops.sm121 import gemm_ar

    fused = _FakeGemmFused()
    monkeypatch.setattr(gemm_ar, "tp_fused_collectives", lambda: fused)
    backend = gemm_ar.Sm121GemmAr(7168)
    linear = _FakeLinear(torch.zeros(7168, 768, dtype=torch.bfloat16))
    decode = torch.zeros(6, 768, dtype=torch.bfloat16)
    assert backend.should_run(decode)
    assert backend.apply(decode, linear).shape == (6, 7168)
    assert fused.launched == [((6, 768), (7168, 768))] and linear.calls == 0
    # Prefill-sized batches and other dtypes keep the projection's own forward.
    assert not backend.should_run(torch.zeros(128, 768, dtype=torch.bfloat16))
    assert not backend.should_run(decode.float())


def test_sm121_gemm_ar_without_fused_collectives_or_geometry(monkeypatch):
    from vllm.models.kimi_k3.nvidia.ops.sm121 import gemm_ar

    backend = gemm_ar.Sm121GemmAr(7168)
    linear = _FakeLinear(torch.zeros(7168, 768, dtype=torch.bfloat16))
    decode = torch.zeros(6, 768, dtype=torch.bfloat16)
    monkeypatch.setattr(gemm_ar, "tp_fused_collectives", lambda: None)
    assert not backend.should_run(decode)
    # A shape the kernel does not cover falls back identically on every rank.
    fused = _FakeGemmFused(eligible=False)
    monkeypatch.setattr(gemm_ar, "tp_fused_collectives", lambda: fused)
    assert backend.should_run(decode)
    backend.apply(decode, linear)
    assert linear.calls == 1 and not fused.launched
