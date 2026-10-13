# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test-owned red zones and byte-exact snapshots; no attention implementation."""

import torch

GUARD_ELEMENTS = 128
CANARY = 73
SENTINEL = -97


class TensorSnapshot:
    def __init__(self, tensor, name):
        self.tensor = tensor
        self.name = name
        self.geometry = self._geometry()
        self.value = tensor.detach().clone()

    def _geometry(self):
        t = self.tensor
        return (t.shape, t.stride(), t.dtype, t.device, t.data_ptr())

    def check(self):
        assert self._geometry() == self.geometry, (
            f"{self.name}: tensor geometry changed"
        )
        # Byte comparison also distinguishes signed zero and NaN payloads.
        current = self.tensor.detach().contiguous().view(torch.uint8)
        saved = self.value.contiguous().view(torch.uint8)
        assert torch.equal(current, saved), f"{self.name}: input bytes changed"


class GuardedTensor:
    def __init__(self, like, name, *, source=False):
        self.name = name
        self.storage = torch.full(
            (like.numel() + 2 * GUARD_ELEMENTS,),
            CANARY,
            dtype=like.dtype,
            device=like.device,
        )
        self.tensor = self.storage[GUARD_ELEMENTS:-GUARD_ELEMENTS].view(like.shape)
        assert self.tensor.is_contiguous()
        if source:
            self.tensor.copy_(like)
        else:
            self.tensor.fill_(SENTINEL)
        self.snapshot = TensorSnapshot(self.tensor, name) if source else None
        self.left = self.storage[:GUARD_ELEMENTS].clone()
        self.right = self.storage[-GUARD_ELEMENTS:].clone()

    def check(self, *, complete=False):
        assert torch.equal(self.storage[:GUARD_ELEMENTS], self.left), (
            f"{self.name}: prefix canary"
        )
        assert torch.equal(self.storage[-GUARD_ELEMENTS:], self.right), (
            f"{self.name}: suffix canary"
        )
        if self.snapshot is not None:
            self.snapshot.check()
        if complete:
            assert self.snapshot is None, "completion applies only to output buffers"
            assert torch.isfinite(self.tensor).all().item(), (
                f"{self.name}: nonfinite output"
            )
            assert (self.tensor != SENTINEL).all().item(), (
                f"{self.name}: unwritten sentinel"
            )


class GuardedEmptyLike:
    """Replace exactly the actual caller's empty_like(q), preserving its layout."""

    def __init__(self, like):
        self.like = like
        self.original = torch.empty_like
        self.output = GuardedTensor(like, "caller-output")
        self.calls = 0

    def __call__(self, tensor, *args, **kwargs):
        if tensor is not self.like:
            return self.original(tensor, *args, **kwargs)
        assert not args and not kwargs, "actual caller allocation contract changed"
        self.calls += 1
        assert self.calls == 1, "actual caller allocated output more than once"
        return self.output.tensor
