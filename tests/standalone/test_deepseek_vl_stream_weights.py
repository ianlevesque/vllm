# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU execution of PR58857's real wrappers, mapper and AutoWeightsLoader.

CUDA model construction/imports and quantized-reload decoration are outside this
gate. Production method bodies, tensor copies and iterator ownership are real.
The unchanged upstream CUDA-platform regression remains a separate native gate.
"""

import dataclasses
import itertools
import logging
import os
import weakref
from pathlib import Path

import pytest
import torch
from test_shared_source_carries import ROOT, load
from torch import nn

UTILS = "vllm/model_executor/models/utils.py"
V4 = "vllm/models/deepseek_v4/common/vl_model.py"
V41 = "vllm/models/deepseek_v41/nvidia/vl_model.py"
UPSTREAM = "tests/models/test_deepseek_v4_vl_stream_weights.py"


def cpu_components():
    logger = logging.getLogger(__name__)
    mapper = load(
        UTILS,
        "WeightsMapper",
        dataclass=dataclasses.dataclass,
        field=dataclasses.field,
        replace=dataclasses.replace,
        logger=logger,
    )
    missing = type("NotUsedMissingLayer", (nn.Module,), {})
    tied = load(UTILS, "_get_tied_embedding_params", VocabParallelEmbedding=missing)
    loader = load(
        UTILS,
        "AutoWeightsLoader",
        nn=nn,
        WeightsMapper=mapper,
        itertools=itertools,
        logger=logger,
        _get_tied_embedding_params=tied,
        StageMissingLayer=missing,
        PPMissingLayer=missing,
        maybe_prefix=load(UTILS, "maybe_prefix"),
        default_weight_loader=load(
            "vllm/model_executor/model_loader/weight_utils.py", "default_weight_loader"
        ),
        support_quantized_model_reload_from_hp_weights=lambda fn: fn,
    )
    return mapper, loader


def wrapper(kind):
    mapper, loader = cpu_components()
    source, name = (
        (V4, "DeepseekV4ForConditionalGeneration")
        if kind == "v4"
        else (V41, "DeepseekV41ForCausalLM")
    )
    # A baseline-negative run changes only the two wrapper source files.
    root = Path(os.environ.get("VLLM_STREAM_WRAPPER_ROOT", ROOT))
    method = load(
        source,
        name + ".load_weights",
        source_text=(root / source).read_text(),
        AutoWeightsLoader=loader,
        stream_language_model_first=load(V4, "stream_language_model_first"),
    )
    return type(name, (nn.Module,), {"load_weights": method}), mapper


@pytest.mark.parametrize("kind", ["v4", "v41"])
def test_unchanged_upstream_streaming_and_single_finalization_assertions(kind):
    cls, mapper = wrapper(kind)
    fake = load(UPSTREAM, "_FakeFinalizingLanguageModel", nn=nn)
    test = load(
        UPSTREAM,
        "test_load_weights_streams_language_model_group",
        nn=nn,
        WeightsMapper=mapper,
        _FakeFinalizingLanguageModel=fake,
    )
    test(lambda: cls)


@pytest.mark.parametrize("kind", ["v4", "v41"])
def test_real_wrapper_maps_names_and_owns_deferred_reused_buffers(kind):
    cls, mapper = wrapper(kind)
    model = cls()
    model.language_model = load(UPSTREAM, "_FakeFinalizingLanguageModel", nn=nn)()
    model.vision = nn.Module()
    model.vision.tensor_b = nn.Parameter(torch.zeros(1))
    model.hf_to_vllm_mapper = mapper(orig_to_new_prefix={"hf.": "language_model."})
    storage = torch.zeros(1)

    def reused():
        storage.fill_(2)
        yield "vision.tensor_b", storage
        storage.fill_(1)
        yield "hf.tensor_a", storage
        assert model.language_model.tensor_a.item() == 1
        storage.fill_(3)
        yield "hf.tensor_c", storage
        storage.fill_(99)

    loaded = model.load_weights(reused())
    assert loaded == {
        "vision.tensor_b",
        "language_model.tensor_a",
        "language_model.tensor_c",
    }
    assert model.vision.tensor_b.item() == 2
    assert model.language_model.finalized_values == [(1, 3)]
    assert model.language_model.load_calls == 1
    assert model._weights_finalized is True


def test_stream_does_not_retain_completed_language_tensors():
    stream = load(V4, "stream_language_model_first")
    refs = []

    def weights():
        for i in range(32):
            tensor = torch.full((8,), float(i))
            refs.append(weakref.ref(tensor))
            yield f"language_model.layer{i}", tensor

    for i, (name, tensor) in enumerate(stream(weights())):
        assert name == f"language_model.layer{i}"
        torch.testing.assert_close(tensor, torch.full((8,), float(i)), rtol=0, atol=0)
        # Current producer/consumer references are allowed; earlier objects are not.
        assert sum(ref() is not None for ref in refs) <= 2
    del tensor
    assert all(ref() is None for ref in refs)


def test_nonfinalizing_v4_child_keeps_original_deferred_finalize_contract():
    cls, mapper = wrapper("v4")
    model = cls()
    model.language_model = load(UPSTREAM, "_FakeFinalizingLanguageModel", nn=nn)()
    model.language_model.finalizes_weights_during_load = False
    model.hf_to_vllm_mapper = mapper()
    loaded = model.load_weights(
        iter(
            [
                ("language_model.tensor_a", torch.tensor([1.0])),
                ("language_model.tensor_c", torch.tensor([3.0])),
            ]
        )
    )
    assert loaded == {"language_model.tensor_a", "language_model.tensor_c"}
    assert model._weights_finalized is False
    assert model.language_model.finalized_values == [(1, 3)]
