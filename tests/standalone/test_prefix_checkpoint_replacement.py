# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU gates for main's replacement of the old K3 prefix LCM workaround.

These execute the real checkpoint planner, reservation predicate and lookup.
CUDA state materialization and complete scheduler replay require model tests.
"""

from collections.abc import Sequence
from types import SimpleNamespace as NS
from typing import overload

import pytest
from test_shared_source_carries import load

INTERFACE = "vllm/v1/kv_cache_interface.py"
UTILS = "vllm/v1/core/kv_cache_utils.py"
MANAGERS = "vllm/v1/core/single_type_kv_cache_manager.py"


class MambaSpec(NS):
    pass


def checkpoint_functions():
    cdiv = lambda a, b: (a + b - 1) // b
    position = load(INTERFACE, "get_mamba_prefill_checkpoint_position")
    valid = load(INTERFACE, "is_mamba_prefill_checkpoint_valid", cdiv=cdiv)
    compute = load(
        "vllm/model_executor/layers/mamba/checkpoint.py",
        "compute_mamba_prefill_checkpoints",
        cdiv=cdiv,
        get_mamba_prefill_checkpoint_position=position,
        is_mamba_prefill_checkpoint_valid=valid,
    )
    reserve = load(
        MANAGERS,
        "MambaManager._needs_internal_checkpoint",
        cdiv=cdiv,
        MambaSpec=MambaSpec,
        is_mamba_prefill_checkpoint_valid=valid,
    )
    return position, compute, reserve


@pytest.mark.parametrize("prompt", [8192, 9216, 10240, 11264, 12288])
@pytest.mark.parametrize("eagle", [False, True])
def test_k3_checkpoint_publication_and_lookup_share_actual_partial_boundary(
    prompt, eagle
):
    position, compute, reserve = checkpoint_functions()
    boundary = position(prompt, 1024, eagle)
    offsets, cols = compute([prompt], [prompt], 1024, 4096, 64, eagle, 1024)
    assert offsets == [boundary]
    assert cols == [(prompt + 4095) // 4096 - 2]
    spec = MambaSpec(block_size=4096, prefill_checkpoint_alignment=64)
    owner = NS(
        kv_cache_spec=spec,
        block_size=4096,
        block_pool=NS(hash_block_size=1024),
        req_to_blocks={"request": []},
        has_prefill_checkpoint_blocks=True,
        _allocated_block_reqs=set(),
        num_speculative_blocks=1,
    )
    assert reserve(owner, "request", 0, prompt, boundary, prompt)
    block_view = load(UTILS, "BlockHashListWithBlockSize", overload=overload)
    resolve = load(UTILS, "resolve_block_hashes", BlockHashListWithBlockSize=block_view)
    lookup = load(
        MANAGERS,
        "MambaManager.find_longest_cache_hit",
        MambaSpec=MambaSpec,
        resolve_block_hashes=resolve,
        Sequence=Sequence,
    )
    checkpoint = object()
    pool = NS(
        hash_block_size=1024,
        null_block=None,
        get_cached_block=lambda key, groups: [checkpoint] if key == boundary else None,
    )
    blocks, hit = lookup(
        NS(supports_fine_grained_hash_lookup=True),
        list(range(1024, prompt + 1, 1024)),
        prompt - 1,
        [1],
        pool,
        spec,
        eagle,
        1024,
        1,
        1,
    )
    assert hit == boundary
    assert blocks[0][-1] is checkpoint
    # 9216/10240/11264 restores are valid when a real snapshot exists;
    # the obsolete LCM carry would discard them solely for being partial.
    if boundary % 4096:
        assert hit > boundary // 4096 * 4096


@pytest.mark.parametrize("alignment", [None, 2048])
def test_worker_and_manager_decline_unmaterializable_checkpoint_together(alignment):
    position, compute, reserve = checkpoint_functions()
    prompt = 8192
    boundary = position(prompt, 1024, True)  # 7168, not a 2048-step export.
    offsets, cols = compute([prompt], [prompt], 1024, 4096, alignment, True, 1024)
    assert offsets == [0] and cols == [-1]
    owner = NS(
        kv_cache_spec=MambaSpec(
            block_size=4096, prefill_checkpoint_alignment=alignment
        ),
        block_size=4096,
        block_pool=NS(hash_block_size=1024),
        req_to_blocks={"request": []},
        has_prefill_checkpoint_blocks=True,
        _allocated_block_reqs=set(),
        num_speculative_blocks=1,
    )
    assert not reserve(owner, "request", 0, prompt, boundary, prompt)
