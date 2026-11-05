# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An explicit --mamba-block-size is the recurrent checkpoint cadence in align
mode (local-inference-lab #457), and the scheduler splits prefill on it.

CPU only: the platform rule runs on stubs, the split is the real
`Scheduler._mamba_block_aligned_split` on a stub self.
"""

import contextlib
import types

import pytest
import torch

from vllm.config.cache import CacheConfig
from vllm.v1.core.sched.scheduler import Scheduler

LATENT = 576
# One fp8 MLA row per token: a 1024-token page holds this state exactly.
MAMBA_PAGE = 1024 * LATENT


def _align(monkeypatch, *, block_size, mamba_block_size, user_block: bool):
    import vllm.config.vllm as vllm_config_mod
    import vllm.model_executor.layers.mamba.abstract as mamba_abstract
    from vllm.model_executor.models import ModelRegistry
    from vllm.platforms.interface import Platform

    model_cls = types.SimpleNamespace(
        get_mamba_specs_from_config=lambda _: [
            types.SimpleNamespace(page_size_bytes=MAMBA_PAGE)
        ]
    )
    monkeypatch.setattr(
        ModelRegistry, "resolve_model_cls", lambda *a, **k: (model_cls, None)
    )
    monkeypatch.setattr(
        vllm_config_mod, "set_current_vllm_config", lambda _: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        mamba_abstract, "get_mamba_prefill_checkpoint_alignment", lambda _: None
    )
    cache = types.SimpleNamespace(
        cache_dtype="fp8",
        block_size=block_size,
        user_specified_block_size=user_block,
        mamba_block_size=mamba_block_size,
        mamba_cache_mode="align",
        mamba_page_size_padded=None,
        kv_cache_dtype_skip_layers=None,
    )
    config = types.SimpleNamespace(
        cache_config=cache,
        parallel_config=None,
        model_config=types.SimpleNamespace(
            dtype=torch.bfloat16,
            use_mla=True,
            architecture="KimiK3ForConditionalGeneration",
            get_num_kv_heads=lambda _: 1,
            get_head_size=lambda: LATENT,
        ),
    )
    backend = types.SimpleNamespace(get_supported_kernel_block_sizes=lambda: [64])
    monkeypatch.setattr(
        Platform, "_find_non_ssm_backends", classmethod(lambda cls, _: [backend])
    )
    Platform._align_hybrid_block_size(config, backend)
    return cache


def test_explicit_cadence_survives_alignment(monkeypatch):
    """Kimi K3 / K3.1 at DCP4: --block-size 1024 --mamba-block-size 4096."""
    cache = _align(monkeypatch, block_size=1024, mamba_block_size=4096, user_block=True)
    assert (cache.block_size, cache.mamba_block_size) == (1024, 4096)


def test_derived_default_follows_a_raised_block(monkeypatch):
    """Without --mamba-block-size the config hook copies --block-size; when the
    platform raises the block, the recurrent grid must follow it."""
    cache = _align(monkeypatch, block_size=256, mamba_block_size=256, user_block=True)
    assert cache.block_size == 1024
    assert cache.mamba_block_size == 1024


def test_derived_default_without_block_size_follows_the_block(monkeypatch):
    default = CacheConfig.DEFAULT_BLOCK_SIZE
    cache = _align(
        monkeypatch, block_size=default, mamba_block_size=default, user_block=False
    )
    assert cache.block_size == 1024
    assert cache.mamba_block_size == 1024


def test_second_alignment_pass_is_stable(monkeypatch):
    """The platform hook can run again on an already aligned config."""
    cache = _align(
        monkeypatch, block_size=1024, mamba_block_size=1024, user_block=False
    )
    assert (cache.block_size, cache.mamba_block_size) == (1024, 1024)


@pytest.mark.parametrize("cadence", [512, 1536])
def test_cadence_off_the_block_grid_falls_back(monkeypatch, cadence):
    cache = _align(
        monkeypatch, block_size=1024, mamba_block_size=cadence, user_block=True
    )
    assert cache.mamba_block_size == cache.block_size == 1024


def _split(num_prompt_tokens, start, budget, *, cadence=None, cache_block=1024):
    request = types.SimpleNamespace(
        num_computed_tokens=start,
        num_prompt_tokens=num_prompt_tokens,
        num_tokens=num_prompt_tokens,
        shared_prefix_boundary=0,
    )
    stub = types.SimpleNamespace(
        block_size=4096,
        cache_config=types.SimpleNamespace(block_size=cache_block),
        use_eagle_block_drop=False,
        num_hash_lookahead_tokens=0,
        max_num_scheduled_tokens=budget,
        scheduler_config=types.SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=False,
        mamba_shared_prefix_checkpoint=False,
        hash_block_size=1024,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_prefill_checkpoint_alignment=None,
    )
    if cadence is not None:
        stub.mamba_block_size = cadence
    return Scheduler._mamba_block_aligned_split(stub, request, budget)


def test_split_lands_on_the_recurrent_grid():
    """A 4100-token budget (K3.1: 4096 + four draft slots) is one cadence."""
    # State slot p holds the state after (p + 1) * 4096 tokens, so no chunk may
    # end on the 1024-token attention grid in between.
    assert _split(20000, 0, 4100, cadence=4096) == 4096
    assert _split(20000, 4096, 4100, cadence=4096) == 4096
    # A chunk that starts mid-block stops at the next recurrent boundary.
    assert _split(20000, 5120, 4100, cadence=4096) == 8192 - 5120


def test_split_without_a_resolved_cadence_keeps_the_cache_block():
    assert _split(20000, 0, 4100) == 4096
    assert _split(20000, 0, 2500) == 2048
    # The old behaviour with a 1024-token cache block and a 4096 Mamba block:
    # the chunk ends between recurrent boundaries.
    assert _split(20000, 0, 2500, cadence=4096) == 2500


def _mamba(block_size: int):
    from vllm.v1.kv_cache_interface import MambaSpec

    return MambaSpec(
        shapes=((8, 16), (2, 8, 8)),
        dtypes=(torch.bfloat16, torch.float32),
        block_size=block_size,
        mamba_cache_mode="align",
    )


def _attention(block_size: int):
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    return FullAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )


def test_scheduler_reads_the_cadence_through_wrapped_mamba_groups():
    """The engine core hands the scheduler unwrapped group specs today; the
    lookup also reads a wrapped group, as KVCacheConfig.has_mamba_layers does."""
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        UniformTypeKVCacheSpecs,
    )

    wrapped = UniformTypeKVCacheSpecs.from_specs(
        {f"kda.{i}": _mamba(4096) for i in range(3)}
    )
    assert isinstance(wrapped, UniformTypeKVCacheSpecs)
    config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn.0"], _attention(1024)),
            KVCacheGroupSpec(list(wrapped.kv_cache_specs), wrapped),
            KVCacheGroupSpec(["kda.9"], _mamba(4096)),
        ],
    )
    assert Scheduler._resolve_mamba_block_size(config, 1024) == 4096

    attention_only = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["attn.0"], _attention(1024))],
    )
    assert Scheduler._resolve_mamba_block_size(attention_only, 1024) == 1024
