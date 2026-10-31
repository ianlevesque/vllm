# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the EXL3 quantization config and extent planning."""

import pytest
import torch

from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.exl3 import (
    Exl3Config,
    plan_exl3_extent,
)


def test_exl3_registered():
    assert get_quantization_config("exl3") is Exl3Config
    assert Exl3Config.get_name() == "exl3"
    assert Exl3Config.get_supported_act_dtypes() == [torch.bfloat16]


def test_from_config_defaults_and_validation():
    cfg = Exl3Config.from_config(
        {"quant_method": "exl3", "exl3": {"codec": "EXL3_K2_LUT_E4M3"}}
    )
    assert cfg.manifest == "exl3-manifest.json"
    with pytest.raises(ValueError, match="manifest"):
        Exl3Config.from_config({"exl3": {"manifest": "elsewhere.json"}})
    with pytest.raises(ValueError, match="codec"):
        Exl3Config.from_config({"exl3": {"codec": "IQ2_XXS"}})


def test_plan_extent_tp8():
    # Kimi K3: 96 slots of 32 channels, TP8 -> 12 slots (384 channels).
    assert plan_exl3_extent(96, 32, 8, 0, 384) == (0, 12)
    assert plan_exl3_extent(96, 32, 8, 7, 384) == (84, 12)
    assert plan_exl3_extent(96, 32, 16, 15, 192) == (90, 6)
    assert plan_exl3_extent(96, 32, 1, 0, 3072) == (0, 96)


def test_plan_extent_rejects_ragged():
    with pytest.raises(ValueError, match="multiple of"):
        plan_exl3_extent(96, 32, 8, 0, 400)
    with pytest.raises(ValueError, match="split evenly"):
        plan_exl3_extent(96, 32, 7, 0, 416)
