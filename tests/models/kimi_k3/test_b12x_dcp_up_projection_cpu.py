# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The decode output of a backend that merges its own DCP partials is up-projected.

`W_UV_dcp` is registered whenever the A2A DCP merge is configured. The
generic merge consumes it and writes the up-projected output itself; B12X_MLA
(`handles_dcp_query_and_output`) skips that merge and returns local heads.
Gating the ordinary up-projection on `W_UV_dcp is None` alone left B12X's
decode output unwritten: the first token (prefill) was right and every
decoded token was noise.
"""

import ast
import re
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "vllm/model_executor/layers/attention/mla_attention.py"
)


def _helper():
    tree = ast.parse(SOURCE.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_generic_dcp_merge_up_projects"
    )
    ns: dict = {}
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), ns)
    return ns["_generic_dcp_merge_up_projects"]


@pytest.mark.parametrize(
    "dcp_world_size,impl_handles_dcp,has_weights,merge_up_projects",
    [
        # No DCP: the ordinary up-projection always runs.
        (1, False, False, False),
        (1, True, False, False),
        # Generic A2A merge with the gathered weights: it up-projects.
        (8, False, True, True),
        # Generic merge without them (all-gather/reduce-scatter, or PCP).
        (8, False, False, False),
        # B12X_MLA under A2A: weights exist, the generic merge is skipped.
        (8, True, True, False),
        (16, True, True, False),
        (8, True, False, False),
    ],
)
def test_who_up_projects(
    dcp_world_size, impl_handles_dcp, has_weights, merge_up_projects
):
    assert (
        _helper()(dcp_world_size, impl_handles_dcp, has_weights) is merge_up_projects
    )


def test_the_decode_path_asks_the_helper():
    source = SOURCE.read_text()
    assert not re.search(r"if self\.W_UV_dcp is None:\s+self\._v_up_proj\(", source)
    assert re.search(
        r"if not _generic_dcp_merge_up_projects\(\s+self\.impl\.dcp_world_size, "
        r"impl_handles_dcp, self\.W_UV_dcp is not None\s+\):\s+"
        r"self\._v_up_proj\(attn_out, out=mqa_output_slice\)",
        source,
    )
