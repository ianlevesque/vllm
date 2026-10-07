# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loading-group lifecycle contracts, runnable without a CUDA installation."""

import importlib.util
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE = (
    Path(__file__).resolve().parents[4]
    / "vllm/model_executor/model_loader/instanttensor_utils.py"
)
spec = importlib.util.spec_from_file_location("instanttensor_utils_contract", MODULE)
assert spec is not None and spec.loader is not None
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


@pytest.fixture
def groups(monkeypatch):
    monkeypatch.delenv("VLLM_INSTANTTENSOR_NCCL_CTAS", raising=False)
    world = SimpleNamespace(
        world_size=16,
        ranks=list(range(16)),
        device_group=SimpleNamespace(
            options=SimpleNamespace(_timeout=timedelta(seconds=90))
        ),
    )
    private = object()
    created, destroyed = [], []
    default = SimpleNamespace(bound_device_id=None)

    def new_group(ranks, **kwargs):
        created.append((ranks, kwargs))
        return private

    dist = SimpleNamespace(
        group=SimpleNamespace(WORLD=default),
        ProcessGroupNCCL=SimpleNamespace(
            Options=lambda: SimpleNamespace(config=SimpleNamespace())
        ),
        new_group=new_group,
        destroy_process_group=destroyed.append,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(distributed=dist))
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    return world, private, default, created, destroyed


def test_default_reuses_but_does_not_destroy_inference_group(groups):
    world, _, _, created, destroyed = groups
    with utils.instanttensor_loading_group(world) as group:
        assert group is world.device_group
    assert created == destroyed == []


def test_loading_group_has_independent_options_and_same_membership(groups, monkeypatch):
    world, private, _, created, destroyed = groups
    monkeypatch.setenv("VLLM_INSTANTTENSOR_NCCL_CTAS", "8")
    before = dict(world.device_group.options.__dict__)
    with utils.instanttensor_loading_group(world) as group:
        assert group is private
        assert destroyed == []
    assert destroyed == [private]
    ranks, options = created[0]
    assert ranks == list(range(16))
    assert options["backend"] == "nccl"
    assert options["pg_options"].config.min_ctas == 8
    assert options["pg_options"].config.max_ctas == 8
    assert options["timeout"] == timedelta(seconds=90)
    assert world.device_group.options.__dict__ == before


def test_private_group_is_destroyed_when_loading_fails(groups, monkeypatch):
    world, private, _, _, destroyed = groups
    monkeypatch.setenv("VLLM_INSTANTTENSOR_NCCL_CTAS", "8")
    with (
        pytest.raises(RuntimeError, match="loading failed"),
        utils.instanttensor_loading_group(world),
    ):
        raise RuntimeError("loading failed")
    assert destroyed == [private]


def test_private_group_is_destroyed_on_early_generator_close(groups, monkeypatch):
    world, private, _, _, destroyed = groups
    monkeypatch.setenv("VLLM_INSTANTTENSOR_NCCL_CTAS", "8")

    def stream():
        with utils.instanttensor_loading_group(world):
            yield "weight"
            raise AssertionError("consumer should close before advancing")

    iterator = stream()
    assert next(iterator) == "weight"
    iterator.close()
    assert destroyed == [private]


@pytest.mark.parametrize("setting", ["", "0", "33", "8.0", "eight"])
def test_invalid_channel_setting_fails_before_creating_a_group(
    groups, monkeypatch, setting
):
    world, _, _, created, destroyed = groups
    monkeypatch.setenv("VLLM_INSTANTTENSOR_NCCL_CTAS", setting)
    with (
        pytest.raises(ValueError, match="integer in 1..32"),
        utils.instanttensor_loading_group(world),
    ):
        pass
    assert created == destroyed == []


def test_bound_parent_is_rejected_instead_of_silently_capping_loading_channels(
    groups, monkeypatch
):
    world, _, default, created, destroyed = groups
    default.bound_device_id = object()
    monkeypatch.setenv("VLLM_INSTANTTENSOR_NCCL_CTAS", "8")
    with (
        pytest.raises(ValueError, match="unbound default group"),
        utils.instanttensor_loading_group(world),
    ):
        pass
    assert created == destroyed == []


@pytest.mark.parametrize("world", [None, SimpleNamespace(world_size=1)])
def test_single_rank_loading_does_not_create_a_collective(groups, monkeypatch, world):
    _, _, _, created, destroyed = groups
    monkeypatch.setenv("VLLM_INSTANTTENSOR_NCCL_CTAS", "8")
    with utils.instanttensor_loading_group(world) as group:
        assert group is None
    assert created == destroyed == []
