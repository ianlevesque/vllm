# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A windowed MLA draft must rebuild its decode metadata each step.

`TritonMLAMetadataBuilder.update_draft_decode_metadata` is a no-op, which is
only sound while the decode metadata is a view of buffers the speculator
refreshes. The retained draft window (`_compact_non_causal_mla_window`) is
gathered into builder-owned copies by `build()`, so a FULL draft graph that
skipped the build would replay its capture batch's table. The builder must
also see the window through the `UniformTypeKVCacheSpecs` wrapper that hybrid
grouping puts around a draft group's spec.

Mistral Large 4's DSpark draft (TP16, target DCP 2, replicated windowed draft
cache) accepted about 5% of its tokens without these two rules and 54% on the
image before the regression.

The sources are parsed rather than imported so this runs without CUDA
extensions.
"""

import ast
import os
from pathlib import Path

import torch

ROOT = Path(os.environ.get("VLLM_SOURCE_ROOT") or Path(__file__).resolve().parents[3])
TRITON_MLA = "vllm/v1/attention/backends/mla/triton_mla.py"
MLA_ATTENTION = "vllm/model_executor/layers/attention/mla_attention.py"


def _class_init(relative: str, class_name: str) -> ast.FunctionDef:
    tree = ast.parse((ROOT / relative).read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )


def _assigned(init: ast.FunctionDef, target: str) -> list[str]:
    return [
        ast.unparse(n.value)
        for n in ast.walk(init)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == target
    ]


def _load_function(relative: str, name: str):
    path = ROOT / relative
    nodes = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    assert len(nodes) == 1
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    namespace: dict = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def test_windowed_draft_does_not_claim_in_place_metadata_updates():
    init = _class_init(TRITON_MLA, "TritonMLAMetadataBuilder")
    assert _assigned(init, "self.supports_draft_decode_metadata_update") == [
        "self.dcp_world_size == 1 and (not self._non_causal_mla_window)"
    ]


def test_window_is_read_through_the_uniform_type_wrapper():
    init = _class_init(MLA_ATTENTION, "MLACommonMetadataBuilder")
    expected = (
        "kv_cache_spec.first_spec "
        "if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs) else kv_cache_spec"
    )
    assert _assigned(init, "_window_spec") == [expected]
    (window,) = _assigned(init, "self._non_causal_mla_window")
    assert window.startswith(
        "_window_spec.sliding_window if isinstance(_window_spec, SlidingWindowMLASpec)"
    )


def test_retained_window_outputs_are_copies_that_need_a_rebuild():
    compact = _load_function(MLA_ATTENTION, "_compact_non_causal_mla_window")
    block_size, window, width = 64, 2051, 48
    columns = torch.arange(width, dtype=torch.int64)
    indices = torch.empty((1, width), dtype=torch.int64)
    shifts = torch.empty(1, dtype=torch.int32)
    compact_table = torch.empty((1, width), dtype=torch.int32)
    compact_lens = torch.empty(1, dtype=torch.int32)

    def run(block_table, seq_lens):
        compact(
            block_table,
            seq_lens,
            window=window,
            block_size=block_size,
            columns=columns,
            indices=indices,
            shifts=shifts,
            compact_table=compact_table,
            compact_lens=compact_lens,
        )

    # The speculator's persistent buffers, as a capture batch leaves them.
    block_table = torch.zeros((1, width), dtype=torch.int32)
    seq_lens = torch.tensor([3], dtype=torch.int32)
    run(block_table, seq_lens)
    assert compact_table.eq(0).all() and compact_lens.item() == 3

    # A live request: refreshing the buffers alone does not reach the copies.
    block_table.copy_(torch.arange(100, 100 + width, dtype=torch.int32)[None])
    seq_lens.fill_(2500)
    assert compact_table.eq(0).all() and compact_lens.item() == 3

    # Only another build() does. floor((2500 - 2051) / 64) = 7 leading blocks
    # fall wholly below the window.
    run(block_table, seq_lens)
    assert compact_table[0, :3].tolist() == [107, 108, 109]
    assert compact_lens.item() == 2500 - 7 * block_size
