# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the DCP-over-RoCE data-movement routing.

The RoCE runtime itself needs GB10 hardware; these tests replace it with a fake
that concatenates every rank's shard, and check that the communicator's
middle-dim gather and gather-based all-to-all reproduce the reference
collectives bit for bit.
"""

import pytest
import torch

from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

WORLD = 4


class _FakeRoce:
    """All-gather over shards the test registered for every rank."""

    disabled = False

    def __init__(self, shards, max_bytes=2 * 1024 * 1024):
        self.shards = shards
        self.max_bytes = max_bytes
        self.calls = []

    def should_all_gather(self, inp, dim):
        if dim < 0:
            dim += inp.dim()
        nbytes = inp.numel() * inp.element_size()
        return (
            inp.is_contiguous()
            and dim in (0, inp.dim() - 1)
            and 0 < nbytes <= self.max_bytes
        )

    def all_gather(self, inp, dim):
        self.calls.append((tuple(inp.shape), dim))
        return torch.cat(
            [s.reshape(inp.shape) for s in self.shards], dim=dim
        ).contiguous()


def _communicator(rank, roce, dcp=True):
    comm = object.__new__(CudaCommunicator)
    comm.world_size = WORLD
    comm.rank_in_group = rank
    comm.b12x_ar_comm = roce
    comm.use_roce_dcp = dcp
    comm.pynccl_comm = None
    return comm


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((1, 2048, 2), torch.float32),
        ((6, 2048, 2), torch.float32),
        ((6, 6, 576), torch.bfloat16),
    ],
)
def test_middle_dim_gather_matches_cat(shape, dtype):
    shards = [torch.randn(shape).to(dtype) for _ in range(WORLD)]
    reference = torch.cat(shards, dim=1)
    for rank in range(WORLD):
        roce = _FakeRoce(shards)
        out = _communicator(rank, roce).all_gather(shards[rank], dim=1)
        assert out.shape == reference.shape
        assert torch.equal(out, reference)
        assert roce.calls == [((shape[0], shape[1] * shape[2]), 1)]


def test_roce_all_to_all_matches_all_to_all_single():
    tokens, heads, width = 6, 6, 514
    sends = [
        torch.randn(WORLD, tokens, heads, width).to(torch.bfloat16)
        for _ in range(WORLD)
    ]
    for rank in range(WORLD):
        recv = torch.zeros_like(sends[rank])
        assert _communicator(rank, _FakeRoce(sends)).roce_all_to_all(sends[rank], recv)
        for source in range(WORLD):
            # all_to_all_single: chunk `source` of recv is what `source`
            # addressed to us.
            assert torch.equal(recv[source], sends[source][rank])


def test_ineligible_shapes_decline_without_sending():
    tokens, heads, width = 4096, 6, 514  # prefill-sized: above the gather cap
    send = torch.zeros(WORLD, tokens, heads, width, dtype=torch.bfloat16)
    roce = _FakeRoce([send] * WORLD)
    assert not _communicator(0, roce).roce_all_to_all(send, torch.zeros_like(send))
    assert roce.calls == []
    assert not _communicator(0, _FakeRoce([send] * WORLD), dcp=False).roce_all_to_all(
        send[:, :1], torch.zeros_like(send[:, :1])
    )
