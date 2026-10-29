# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Router-gate replication selection for the latent-MoE layer.

``_replicate_router_gate`` decides whether the MoE gate keeps its full
weight on every TP rank (replicated ``GateLinear``) or is sharded by expert
(``KimiColumnParallelGate`` + one fp32-logit all-gather per layer). The
latent projections keep following ``shard_auxiliary_projections`` either
way; only the gate is decoupled, and only under the opt-in env flag.
"""

import pytest

from vllm import envs
from vllm.models.kimi_k3.nvidia.model import _replicate_router_gate

ENV = "VLLM_KIMI_K3_REPLICATE_ROUTER_GATE"


@pytest.mark.parametrize("shard_auxiliary", [True, False])
def test_replicate_router_gate_default(monkeypatch, shard_auxiliary):
    monkeypatch.delenv(ENV, raising=False)
    # Feature TP sharding keeps the sharded gate unless opted in; without
    # auxiliary sharding the gate is always replicated.
    assert _replicate_router_gate(shard_auxiliary) == (not shard_auxiliary)
    assert envs.VLLM_KIMI_K3_REPLICATE_ROUTER_GATE is False


@pytest.mark.parametrize("shard_auxiliary", [True, False])
def test_replicate_router_gate_opt_in(monkeypatch, shard_auxiliary):
    monkeypatch.setenv(ENV, "1")
    assert _replicate_router_gate(shard_auxiliary) is True


def test_replicate_router_gate_explicit_off(monkeypatch):
    monkeypatch.setenv(ENV, "0")
    assert _replicate_router_gate(True) is False
    assert _replicate_router_gate(False) is True
