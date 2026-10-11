# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU differential and lifecycle gates for the retained bulk loader."""

import json
import struct
import sys
from collections import defaultdict
from collections.abc import Iterable
from types import SimpleNamespace as NS
from typing import cast
from unittest.mock import Mock

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from test_shared_source_carries import load

WEIGHTS = "vllm/model_executor/model_loader/weight_utils.py"
QWEN_MTP = "vllm/models/qwen4_exp/nvidia/mtp.py"
DEFAULT_LOADER = "vllm/model_executor/model_loader/default_loader.py"


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


def qwen_mtp_without_device_parameters(**namespace):
    """Load the real class; only device-allocating construction is bypassed."""
    model_type = load(
        QWEN_MTP,
        "Qwen4ExpMTP",
        nn=torch.nn,
        SupportsPP=type("SupportsPP", (), {}),
        Qwen4ExpMixtureOfExperts=type("Qwen4ExpMixtureOfExperts", (), {}),
        _remap_mtp_weight_name=load(QWEN_MTP, "_remap_mtp_weight_name"),
        **namespace,
    )
    model = model_type.__new__(model_type)
    torch.nn.Module.__init__(model)
    return model


@pytest.mark.parametrize("buffer_size", [1, 32, 4096])
@pytest.mark.parametrize("prefix", ["", "model.language_model.", "language_model."])
def test_qwen_mtp_prefilter_preserves_values_without_reading_target_payload(
    tmp_path, monkeypatch, buffer_size, prefix
):
    """Actual hook -> loader -> selected ranges -> real file bytes on CPU.

    CUDA/native I/O is replaced at InstantTensor's boundary. Shard filtering,
    range selection, oversized fallback, and draft name mapping are real.
    """
    embedding = prefix + "embed_tokens.weight"
    draft = prefix + "mtp.layers.0.mlp.experts.gate_up_proj"
    ple = prefix + "layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
    mixed_ple = prefix + "layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight"
    unused = prefix + "layers.0.mlp.experts.gate_up_proj"
    only_target = str(tmp_path / "target.safetensors")
    mixed = str(tmp_path / "mixed.safetensors")
    fixtures = {
        only_target: {ple: torch.arange(128, dtype=torch.float32)},
        mixed: {
            embedding: torch.tensor([1.0, 2.0]),
            unused: torch.arange(256, dtype=torch.float32),
            mixed_ple: torch.arange(128, dtype=torch.float32),
            draft: torch.arange(16, dtype=torch.float32).reshape(4, 4),
            "lm_head.weight": torch.tensor([3.0, 4.0]),
        },
    }
    for path, weights in fixtures.items():
        save_file(weights, path)
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    name: str(path)
                    for path, weights in fixtures.items()
                    for name in weights
                }
            }
        )
    )
    expected_names = {embedding, draft, "lm_head.weight"}
    expected = {name: fixtures[mixed][name] for name in expected_names}
    opened, copied, fallback, entered = [], [], [], []

    class Reader(NS):
        def __enter__(self):
            entered.append(self)
            return self

        def __exit__(self, *args):
            return False

        def tensors(self):
            # Consume the selected physical ranges, including disjoint runs.
            offset_pos = 0
            for name, metadata in self.ordered_tensor_metadatas:
                file_idx, start = self.tensor_offsets[offset_pos]
                _, end = self.tensor_offsets[offset_pos + 1]
                with open(self.filename[file_idx], "rb") as handle:
                    header_size = struct.unpack("<Q", handle.read(8))[0]
                    handle.seek(8 + header_size + start)
                    raw = bytearray(handle.read(end - start))
                assert len(raw) == end - start
                copied.append(name)
                yield (
                    name,
                    torch.frombuffer(raw, dtype=torch.float32)
                    .reshape(metadata["shape"])
                    .clone(),
                )
                offset_pos += 1
                if (
                    offset_pos + 1 < len(self.tensor_offsets)
                    and self.tensor_offsets[offset_pos + 1][0] != file_idx
                ):
                    offset_pos += 1

    def open_reader(files, **kwargs):
        assert kwargs["load_now"] is False
        assert kwargs["copy"] is True
        opened.extend(files)
        return Reader(**vars(layout(files)))

    def select(reader, **kwargs):
        selected = restrict()(reader, **kwargs)
        fallback.extend(name for name, _ in selected)
        return selected

    monkeypatch.setenv("INSTANTTENSOR_BUFFER_SIZE", str(buffer_size))
    monkeypatch.setitem(sys.modules, "instanttensor", NS(safe_open=open_reader))
    iterator = load(
        WEIGHTS,
        "_instanttensor_weights_iterator",
        current_platform=NS(is_cuda=lambda: True, current_device=lambda: 0),
        safe_open=safe_open,
        defaultdict=defaultdict,
        logger=Mock(),
        _restrict_instanttensor_to_selected_ranges=select,
        tqdm=Progress,
        enable_tqdm=lambda _: False,
        _BAR_FORMAT="",
    )
    method = load(
        DEFAULT_LOADER,
        "DefaultModelLoader._get_weights_iterator",
        json=json,
        filter_safetensors_files_by_weight_name=load(
            WEIGHTS,
            "filter_safetensors_files_by_weight_name",
            safe_open=safe_open,
            logger=Mock(),
        ),
        instanttensor_weights_iterator=iterator,
    )
    owner = NS(
        load_config=NS(
            model_loader_extra_config={},
            load_format="instanttensor",
            use_tqdm_on_load=False,
        ),
        _prepare_weights=lambda *args: (
            str(tmp_path),
            list(fixtures),
            True,
            index.name,
        ),
        _encoder_only_lm_prefixes=None,
        counter_before_loading_weights=1.0,
    )
    owner._get_weights_iterator = lambda source: method(owner, source)
    get_all = load(
        DEFAULT_LOADER,
        "DefaultModelLoader.get_all_weights",
        DefaultModelLoader=NS(Source=load(DEFAULT_LOADER, "DefaultModelLoader.Source")),
        Iterable=Iterable,
        cast=cast,
    )
    actual = dict(
        get_all(
            owner,
            NS(model=str(tmp_path), revision=None),
            qwen_mtp_without_device_parameters(),
        )
    )
    assert set(actual) == expected_names
    assert opened == [mixed]  # Target-only shard never reaches native construction.
    assert set(copied).isdisjoint({ple, mixed_ple, unused})
    assert set(fallback).isdisjoint({ple, mixed_ple, unused})
    assert set(copied) | set(fallback) == expected_names
    assert bool(entered) == (buffer_size != 1)
    assert set(fallback) == {
        name
        for name, value in expected.items()
        if value.numel() * value.element_size() > buffer_size
    }
    for name, expected_value in expected.items():
        torch.testing.assert_close(actual[name], expected_value, rtol=0, atol=0)


@pytest.mark.parametrize("prefix", ["", "model.language_model.", "language_model."])
@pytest.mark.parametrize(
    "name, mapped",
    [
        ("embed_tokens.weight", "model.embed_tokens.weight"),
        ("model.embed_tokens.weight", "model.embed_tokens.weight"),
        ("lm_head.weight", "lm_head.weight"),
        ("model.lm_head.weight", "lm_head.weight"),
        ("shared_head.head.weight", "lm_head.weight"),
        ("model.shared_head.head.weight", "lm_head.weight"),
        ("mtp.shared_head.head.weight", "lm_head.weight"),
        ("mtp.layers.0.weight", "model.layers.0.weight"),
        ("model.mtp.layers.0.weight", "model.layers.0.weight"),
    ],
)
def test_qwen_mtp_prefilter_preserves_existing_alias_consumer_values(
    prefix, name, mapped
):
    received = {}

    class CaptureLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load_weights(self, weights, **kwargs):
            received.update(weights)
            return set(received)

    model = qwen_mtp_without_device_parameters(
        AutoWeightsLoader=CaptureLoader,
        WeightsMapper=lambda **kwargs: None,
        _QWEN4_EXP_IGNORED_MISSING_SUFFIXES=[],
    )
    value = torch.tensor([7.0, 11.0])
    weights = [(prefix + name, value), (prefix + "layers.0.weight", torch.zeros(1))]
    selected = (
        (key, tensor)
        for key, tensor in weights
        if not model.is_unused_checkpoint_weight(key)
    )
    assert model.load_weights(selected) == {mapped}
    assert received[mapped] is value
