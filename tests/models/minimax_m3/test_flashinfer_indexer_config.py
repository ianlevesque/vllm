# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where the FlashInfer indexer's metadata builder reads its local window.

The builder module imports FlashInfer's MSA kernels, so the helper is loaded
from source instead of importing the module.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "vllm/models/minimax_m3/nvidia/indexer_flashinfer.py"
)


def _helper():
    tree = ast.parse(SOURCE.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_index_local_blocks"
    )
    ns: dict = {}
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), ns)
    return ns["_index_local_blocks"]


def test_the_builder_does_not_read_the_dropped_dict_directly():
    # The failure this guards: `text_config.sparse_attention_config` raised
    # AttributeError on every rank once the KV cache was being initialised.
    assert "text_config.sparse_attention_config" not in SOURCE.read_text()


@pytest.mark.parametrize("local_blocks", [0, 1, 3])
def test_upstream_schema_text_config(local_blocks):
    # A checkpoint saved in the transformers schema, e.g.
    # lukealonso/MiniMax-M3-NVFP4: flat `index_*` fields, no dict.
    from transformers import MiniMaxM3VLTextConfig

    config = MiniMaxM3VLTextConfig(index_local_blocks=local_blocks)
    assert not hasattr(config, "sparse_attention_config")
    assert _helper()(config) == local_blocks


def test_legacy_dict_is_folded_by_the_config_class():
    # The original checkpoint's text_config ships the dict; the class folds
    # it into `index_local_blocks` and drops it.
    from transformers import MiniMaxM3VLTextConfig

    config = MiniMaxM3VLTextConfig(
        sparse_attention_config={"sparse_local_block": 2, "sparse_topk_blocks": 16}
    )
    assert not hasattr(config, "sparse_attention_config")
    assert config.index_local_blocks == 2
    assert _helper()(config) == 2


def test_a_config_that_only_carries_the_dict_is_read_the_old_way():
    config = SimpleNamespace(sparse_attention_config={"sparse_local_block": 4})
    assert _helper()(config) == 4
    assert _helper()(SimpleNamespace(sparse_attention_config={})) == 0
    assert _helper()(SimpleNamespace()) == 0
