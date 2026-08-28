# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the production draft override composer without native imports."""

from __future__ import annotations

import ast
import functools
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(os.environ.get("VLLM_CARRY_TEST_ROOT", Path(__file__).resolve().parents[2]))
TREE = ast.parse((ROOT / "vllm/config/speculative.py").read_text())
SOURCE_CLASS = next(
    node
    for node in TREE.body
    if isinstance(node, ast.ClassDef) and node.name == "SpeculativeConfig"
)
METHOD_NAMES = {
    "_apply_composed_hf_override",
    "_update_nested_hf_override",
    "_apply_composed_dict_hf_override",
    "compose_draft_hf_overrides",
}
NAMESPACE = {"functools": functools}
MODULE = ast.Module(
    body=[
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        ast.ClassDef(
            name="SpeculativeConfig",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[
                node
                for node in SOURCE_CLASS.body
                if isinstance(node, ast.FunctionDef) and node.name in METHOD_NAMES
            ],
        ),
    ],
    type_ignores=[],
)
exec(
    compile(ast.fix_missing_locations(MODULE), "<production overrides>", "exec"),
    NAMESPACE,
)
SpeculativeConfig = NAMESPACE["SpeculativeConfig"]


def map_draft_architecture(config):
    config.architectures = ["Glm5NextMTPModel"]
    return config


SpeculativeConfig.hf_config_override = staticmethod(map_draft_architecture)


def compose_from_production_callsite(method, overrides):
    call = next(
        node
        for node in ast.walk(SOURCE_CLASS)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compose_draft_hf_overrides"
    )
    scope = dict(
        NAMESPACE,
        self=SimpleNamespace(
            method=method, target_model_config=SimpleNamespace(hf_overrides=overrides)
        ),
    )
    return eval(compile(ast.Expression(call), "<production callsite>", "eval"), scope)


@pytest.mark.parametrize("nested_dict", [False, True])
def test_mtp_preserves_nested_topk_without_losing_checkpoint_fields(nested_dict):
    text = {"index_topk": 2048, "index_kpool": 4, "hidden_size": 6144}
    if not nested_dict:
        text = SimpleNamespace(**text)
    config = SimpleNamespace(text_config=text, architectures=["Glm5NextForCausalLM"])
    overrides = {"text_config": {"index_topk": 2044}}
    result = compose_from_production_callsite("mtp", overrides)(config)
    values = result.text_config if nested_dict else vars(result.text_config)
    assert values == {"index_topk": 2044, "index_kpool": 4, "hidden_size": 6144}
    assert (
        (values["index_topk"] + values["index_kpool"] - 1 + 127) // 128
    ) * 128 == 2048
    assert result.architectures == ["Glm5NextMTPModel"]
    assert overrides == {"text_config": {"index_topk": 2044}}


@pytest.mark.parametrize("method", ["eagle", "dflash", "dspark"])
def test_separate_draft_keeps_its_own_nested_parameters(method):
    config = SimpleNamespace(text_config=SimpleNamespace(index_topk=2048))
    transform = compose_from_production_callsite(
        method, {"text_config": {"index_topk": 2044}}
    )
    assert transform(config).text_config.index_topk == 2048


@pytest.mark.parametrize("method", ["mtp", "eagle"])
def test_callable_transform_still_runs_after_architecture_mapping(method):
    def override(config):
        assert config.architectures == ["Glm5NextMTPModel"]
        config.text_config.index_topk = 2044
        return config

    config = SimpleNamespace(text_config=SimpleNamespace(index_topk=2048))
    transform = compose_from_production_callsite(method, override)
    assert isinstance(transform, functools.partial)
    assert transform(config).text_config.index_topk == 2044


def test_unset_override_still_maps_draft_architecture():
    config = SimpleNamespace()
    assert compose_from_production_callsite("mtp", None)(config).architectures == [
        "Glm5NextMTPModel"
    ]
