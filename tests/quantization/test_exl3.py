# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the EXL3 quantization config and extent planning."""

import pytest
import torch

from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.exl3 import (
    Exl3Config,
    make_exl3_activation_spec,
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


def test_dense_fp8_config_and_predicate():
    from vllm.model_executor.layers.quantization.exl3 import dense_fp8_applies

    cfg = Exl3Config.from_config(
        {
            "quant_method": "exl3",
            "exl3": {"codec": "EXL3_K2_LUT_E4M3"},
            "dense_fp8": {"scheme": "per_tensor_e4m3"},
        }
    )
    assert cfg.dense_fp8 is True
    plain = Exl3Config.from_config({"exl3": {}})
    assert plain.dense_fp8 is False
    with pytest.raises(ValueError, match="dense_fp8"):
        Exl3Config.from_config({"dense_fp8": {"scheme": "block"}})

    base = "language_model.model.layers.0"
    assert dense_fp8_applies(f"{base}.self_attn.q_proj", 3584, 3584)
    assert dense_fp8_applies(
        f"{base}.block_sparse_moe.shared_experts.gate_up_proj", 3584, 6144
    )
    assert not dense_fp8_applies(
        f"{base}.block_sparse_moe.gate", 3584, 896
    )
    assert not dense_fp8_applies("vision_tower.blocks.0.attn.qkv", 1280, 3840)
    assert not dense_fp8_applies(f"{base}.input_layernorm", 3584, 1)
    assert not dense_fp8_applies("language_model.model.embed_tokens", 3584, 163840)

    mlp_cfg = Exl3Config.from_config(
        {"dense_fp8": {"scheme": "per_tensor_e4m3", "scope": "mlp_only"}}
    )
    assert mlp_cfg.dense_fp8 is True
    assert mlp_cfg.dense_fp8_scope == "mlp_only"
    with pytest.raises(ValueError, match="scope"):
        Exl3Config.from_config(
            {"dense_fp8": {"scheme": "per_tensor_e4m3", "scope": "bogus"}}
        )
    assert not dense_fp8_applies(
        f"{base}.self_attn.q_proj", 3584, 3584, "mlp_only"
    )
    assert not dense_fp8_applies(
        f"{base}.self_attn.f_a_proj", 3584, 3584, "mlp_only"
    )
    assert dense_fp8_applies(
        f"{base}.block_sparse_moe.shared_experts.gate_up_proj",
        3584,
        6144,
        "mlp_only",
    )
    with pytest.raises(ValueError, match="scope"):
        dense_fp8_applies(f"{base}.mlp.up_proj", 3584, 3584, "bogus")


def test_activation_spec_uses_fp16_rotations_with_bf16_io():
    captured = {}

    def fake_activation_spec(**kwargs):
        captured.update(kwargs)
        return captured

    spec = make_exl3_activation_spec(fake_activation_spec, "situ")

    assert spec is captured
    assert captured == {
        "mode": "a16",
        "nonlinearity": "situ",
        "io_dtype": torch.bfloat16,
        "rotation_dtype": torch.float16,
    }
