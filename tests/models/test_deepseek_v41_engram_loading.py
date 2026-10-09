"""CPU byte-oracle regression for bounded integrated-GPU Engram loading."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def loader(integrated=True):
    # Exercise the installed function in isolation: import-time model kernels
    # require a GPU, but weight sharding/staging can be verified byte for byte.
    source = Path(__file__).parents[2] / "vllm/models/deepseek_v41/common/engram.py"
    tree = ast.parse(source.read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_engram_head_shard_weight_loader"
    )
    module = ast.Module(body=[fn], type_ignores=[])
    scope = {
        "torch": torch,
        "current_platform": SimpleNamespace(is_integrated_gpu=lambda _: integrated),
        "logger": SimpleNamespace(info=lambda *args: None),
    }
    exec(compile(module, str(source), "exec"), scope)
    return scope[fn.name]


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e8m0fnu])
def test_tp_slice_and_chunk_tail_preserve_every_byte(dtype, monkeypatch):
    rows, width, offset = 280003, 256, 31
    bits = (torch.arange((rows + offset + 7) * width, dtype=torch.int64) % 251).to(
        torch.uint8
    )
    source = bits.view(dtype).view(-1, width)
    destination = torch.empty(
        rows, width, dtype=torch.uint8 if dtype == torch.float8_e8m0fnu else dtype
    )
    param = SimpleNamespace(
        shape=destination.shape,
        data=destination,
        engram_vocab_start=offset,
        device=SimpleNamespace(type="cuda", index=0),
    )
    clones = []
    original = torch.Tensor.clone

    def capture(tensor, *args, **kwargs):
        clones.append(tensor.numel() * tensor.element_size())
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", capture)
    loader()(param, source)
    assert torch.equal(
        destination.view(torch.uint8), source[offset : offset + rows].view(torch.uint8)
    )
    assert len(clones) == 2
    assert max(clones) <= 64 * 1024 * 1024
    assert sum(clones) == rows * width


@pytest.mark.parametrize("integrated,cpu_destination", [(False, False), (True, True)])
def test_other_copy_paths_keep_direct_copy(integrated, cpu_destination, monkeypatch):
    source = torch.arange(1024, dtype=torch.int32).view(-1, 8)
    destination = torch.empty(101, 8, dtype=torch.int32)
    param = SimpleNamespace(
        shape=destination.shape,
        data=destination,
        engram_vocab_start=13,
        device=SimpleNamespace(type="cpu" if cpu_destination else "cuda", index=0),
    )
    monkeypatch.setattr(
        torch.Tensor, "clone", lambda *_: pytest.fail("unexpected staging")
    )
    loader(integrated)(param, source)
    assert torch.equal(destination, source[13:114])
