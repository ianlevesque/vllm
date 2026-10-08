# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise DSpark's actual draft loader call before allocating weights."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.v1.worker.gpu.spec_decode.dspark import utils


@dataclass
class _AttentionConfig:
    backend: str | None = None
    use_non_causal: bool = False


@dataclass
class _Config:
    speculative_config: object
    attention_config: _AttentionConfig


class _Captured(Exception):
    def __init__(self, config):
        self.config = config


@pytest.mark.parametrize("explicit", [True, False])
def test_dspark_forwards_draft_loader_without_changing_default(explicit):
    requested = object() if explicit else None
    speculative = SimpleNamespace(
        draft_load_config=requested,
        draft_model_config=SimpleNamespace(hf_config=object()),
        attention_backend="TRITON_ATTN",
    )
    cfg = _Config(speculative, _AttentionConfig())

    def capture(*, vllm_config, model_config, load_config=None):
        assert model_config is speculative.draft_model_config
        assert vllm_config.attention_config.backend == "TRITON_ATTN"
        raise _Captured(load_config)

    with (
        patch.object(utils, "_create_draft_vllm_config", return_value=cfg),
        patch(
            "vllm.model_executor.models.qwen3_dflash.dflash_has_any_non_causal",
            return_value=False,
        ),
        patch.object(utils, "get_model", side_effect=capture),
        pytest.raises(_Captured) as caught,
    ):
        utils.load_dspark_model(object(), cfg)

    assert caught.value.config is requested
