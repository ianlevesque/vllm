# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU differential and lifecycle gates for the retained bulk loader."""

import json
import struct
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from test_shared_source_carries import load

WEIGHTS = "vllm/model_executor/model_loader/weight_utils.py"


def layout(files):
    metadata, offsets = [], []
    for i, path in enumerate(files):
        with open(path, "rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(length))
        with safe_open(path, framework="pt") as reader:
            names = list(reader.offset_keys())
        for name in names:
            metadata.append((name, header[name]))
            offsets.append((i, header[name]["data_offsets"][0]))
        offsets.append((i, header[names[-1]]["data_offsets"][1]))
    sizes = [m["data_offsets"][1] - m["data_offsets"][0] for _, m in metadata]
    return NS(
        filename=list(files),
        ordered_tensor_metadatas=metadata,
        tensor_offsets=offsets,
        tensor_sizes=sizes,
        total_tensor_size=sum(sizes),
        tensor_name_to_index={},
        loader_handle=None,
        _determine_buffer_size=Mock(),
    )


def restrict():
    return load(
        WEIGHTS, "_restrict_instanttensor_to_selected_ranges", safe_open=safe_open
    )


@pytest.mark.parametrize("max_size", [None, 8, 1])
@pytest.mark.parametrize("skip_dense", [False, True])
def test_indexed_overlay_selects_correct_physical_owner_before_io(
    tmp_path, max_size, skip_dense
):
    base, overlay = [
        str(tmp_path / x) for x in ("base.safetensors", "overlay.safetensors")
    ]
    save_file(
        {"dense": torch.tensor([1.0]), "expert": torch.tensor([99.0, 99.0])}, base
    )
    save_file({"expert": torch.tensor([3.0, 3.0, 3.0])}, overlay)
    reader = layout([base, overlay])
    fallback = restrict()(
        reader,
        indexed_tensor_files={"dense": base, "expert": overlay},
        is_unused_weight=lambda name: skip_dense and name == "dense",
        max_tensor_size=max_size,
    )
    gpu_names = [name for name, _ in reader.ordered_tensor_metadatas]
    all_names = gpu_names + [name for name, _ in fallback]
    assert sorted(all_names) == (["expert"] if skip_dense else ["dense", "expert"])
    assert sum(reader.tensor_sizes) == reader.total_tensor_size
    assert reader.tensor_name_to_index == {n: i for i, n in enumerate(gpu_names)}
    # The stale base expert must never enter either GPU ranges or CPU fallback.
    for name, path in fallback:
        assert path == (overlay if name == "expert" else base)
    if "expert" in gpu_names:
        assert reader.filename[-1] == overlay
        assert reader.tensor_sizes[gpu_names.index("expert")] == 12
    if gpu_names:
        reader._determine_buffer_size.assert_called_once_with(None)


def test_noncontiguous_selection_emits_disjoint_file_runs(tmp_path):
    path = str(tmp_path / "weights.safetensors")
    save_file({k: torch.ones(2) for k in ("a", "b", "c")}, path)
    reader = layout([path])
    fallback = restrict()(
        reader, indexed_tensor_files=None, is_unused_weight=lambda n: n == "b"
    )
    assert not fallback
    assert reader.filename == [path, path]
    assert [n for n, _ in reader.ordered_tensor_metadatas] == ["a", "c"]
    assert reader.tensor_offsets == [(0, 0), (0, 8), (1, 16), (1, 24)]


@pytest.mark.parametrize("fault", ["opened", "order", "count", "empty"])
def test_incompatible_layout_fails_before_io(tmp_path, fault):
    path = str(tmp_path / "weights.safetensors")
    save_file({"a": torch.ones(2)}, path)
    reader = layout([path])
    if fault == "opened":
        reader.loader_handle = object()
    elif fault == "order":
        reader.ordered_tensor_metadatas[0] = ("wrong", {"data_offsets": [0, 8]})
    elif fault == "count":
        reader.tensor_offsets = []
    with pytest.raises(RuntimeError):
        restrict()(
            reader,
            indexed_tensor_files={} if fault == "empty" else None,
            is_unused_weight=None,
        )


class Progress:
    _get_free_pos = staticmethod(lambda: 0)

    def __init__(self, **kwargs):
        pass

    def update(self, count):
        pass

    def close(self):
        pass


@pytest.mark.parametrize("explicit_copy", [False, True])
def test_current_loader_owns_buffers_with_default_and_enabled_recipe(
    monkeypatch, explicit_copy
):
    if explicit_copy:
        monkeypatch.setenv("INSTANTTENSOR_COPY", "1")
    else:
        monkeypatch.delenv("INSTANTTENSOR_COPY", raising=False)
    monkeypatch.delenv("INSTANTTENSOR_BUFFER_SIZE", raising=False)
    opened = []
    tensor = torch.ones(4)

    class Reader:
        total_tensor_size = 16

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def tensors(self):
            yield "weight", tensor

    def open_reader(files, **kwargs):
        opened.append(kwargs)
        return Reader()

    monkeypatch.setitem(sys.modules, "instanttensor", NS(safe_open=open_reader))
    method = load(
        WEIGHTS,
        "_instanttensor_weights_iterator",
        current_platform=NS(is_cuda=lambda: True, current_device=lambda: 0),
        tqdm=Progress,
        enable_tqdm=lambda _: False,
        _BAR_FORMAT="",
    )
    assert list(method(["weights.safetensors"], False)) == [("weight", tensor)]
    assert opened[0]["copy"] is True
    assert not hasattr(tensor, "_vllm_instanttensor_borrowed")


def test_default_loader_passes_custom_index_and_current_unused_weight_filter(tmp_path):
    index = tmp_path / "custom.index.json"
    index.write_text(json.dumps({"weight_map": {"wanted": "overlay.safetensors"}}))
    captured = {}

    def iterator(files, progress, **kwargs):
        captured.update(kwargs)
        return iter([("wanted", torch.ones(1))])

    skip = lambda name: name == "unused"
    owner = NS(
        load_config=NS(
            model_loader_extra_config={},
            load_format="instanttensor",
            use_tqdm_on_load=False,
        ),
        _prepare_weights=lambda *args: (str(tmp_path), ["shard"], True, index.name),
        _encoder_only_lm_prefixes=None,
        counter_before_loading_weights=1.0,
    )
    source = NS(
        model_or_path="model",
        subfolder=None,
        revision=None,
        fall_back_to_pt=True,
        allow_patterns_overrides=None,
        is_unused_weight=skip,
        prefix="model.",
    )
    method = load(
        "vllm/model_executor/model_loader/default_loader.py",
        "DefaultModelLoader._get_weights_iterator",
        json=json,
        filter_safetensors_files_by_weight_name=lambda files, _: files,
        instanttensor_weights_iterator=iterator,
    )
    result = list(method(owner, source))
    assert result[0][0] == "model.wanted"
    assert captured["indexed_tensor_files"] == {
        "wanted": str(tmp_path / "overlay.safetensors")
    }
    assert captured["is_unused_weight"] is skip


def test_oversized_tensors_bypass_gpu_ring_without_stale_overlay(tmp_path, monkeypatch):
    from collections import defaultdict

    base, overlay = [
        str(tmp_path / x) for x in ("base.safetensors", "overlay.safetensors")
    ]
    save_file(
        {"dense": torch.tensor([1.0]), "expert": torch.tensor([99.0, 99.0])}, base
    )
    save_file({"expert": torch.tensor([3.0, 3.0, 3.0])}, overlay)
    monkeypatch.setenv("INSTANTTENSOR_BUFFER_SIZE", "1")
    reader = layout([base, overlay])
    calls = []

    def open_reader(*args, **kwargs):
        calls.append(kwargs)
        return reader

    monkeypatch.setitem(sys.modules, "instanttensor", NS(safe_open=open_reader))
    method = load(
        WEIGHTS,
        "_instanttensor_weights_iterator",
        current_platform=NS(is_cuda=lambda: True, current_device=lambda: 0),
        safe_open=safe_open,
        defaultdict=defaultdict,
        logger=Mock(),
        _restrict_instanttensor_to_selected_ranges=restrict(),
    )
    weights = dict(
        method(
            [base, overlay],
            False,
            indexed_tensor_files={"dense": base, "expert": overlay},
        )
    )
    torch.testing.assert_close(weights["dense"], torch.tensor([1.0]))
    torch.testing.assert_close(weights["expert"], torch.tensor([3.0, 3.0, 3.0]))
    assert calls[0]["load_now"] is False
    assert calls[0]["copy"] is True
    assert reader.ordered_tensor_metadatas == []


@pytest.mark.parametrize("buffer_size", [1, 32])
def test_real_instanttensor_constructor_handles_cpu_only_and_mixed_selection(
    tmp_path, monkeypatch, buffer_size
):
    """Use installed metadata parsing/buffer sizing; stub only CUDA/native probes.

    The older layout fixture mocks _determine_buffer_size and cannot detect
    InstantTensor's max([]) failure when no GPU tensors remain.
    """
    instanttensor = pytest.importorskip("instanttensor")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (16 << 30, 16 << 30))
    monkeypatch.setattr(instanttensor._C, "file_in_memory", lambda _: False)
    monkeypatch.setattr(instanttensor._C, "backend_available", lambda _: True)
    monkeypatch.setenv("INSTANTTENSOR_BUFFER_SIZE", str(buffer_size))
    path = str(tmp_path / "weights.safetensors")
    save_file({"small": torch.arange(2), "large": torch.arange(40)}, path)
    # This executes the dependency's real safe_open constructor, metadata
    # reader and _determine_buffer_size, with native I/O deliberately unopened.
    reader = instanttensor.safe_open(
        [path], framework="pt", device="cuda:0", load_now=False, copy=True
    )
    assert reader.loader_handle is None
    fallback = restrict()(
        reader,
        indexed_tensor_files={"small": path, "large": path},
        is_unused_weight=None,
        max_tensor_size=buffer_size,
    )
    gpu_names = [name for name, _ in reader.ordered_tensor_metadatas]
    assert gpu_names == ([] if buffer_size == 1 else ["small"])
    assert {name for name, _ in fallback} == (
        {"small", "large"} if buffer_size == 1 else {"large"}
    )
    assert reader.loader_handle is None
    if gpu_names:
        assert reader.buffer_size == 16
