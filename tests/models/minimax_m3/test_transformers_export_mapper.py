# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""transformers-style MiniMax-M3 export names -> this backbone's parameters.

The names are the 34 text-backbone key patterns of lukealonso/MiniMax-M3-NVFP4
(revision e7c7b65d, 176,572 tensors), read from its safetensors headers. The
strict `AutoWeightsLoader` rejects any name that resolves to no module, so
every pattern must either land on this backbone's layout or be dropped.
"""

import pytest

from vllm.models.minimax_m3.nvidia.model import MiniMaxM3SparseForCausalLM

MAPPER = MiniMaxM3SparseForCausalLM.hf_to_vllm_mapper
LM = "model.language_model."
MOE = "model.layers.3.block_sparse_moe."


def mapped(name: str):
    return MAPPER._map_name_with_shard(name)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # The router's score-correction bias: under the gate in the export,
        # on the MoE block here. 57 tensors, bf16 [128].
        (
            LM + "layers.3.mlp.gate.e_score_correction_bias",
            (MOE + "e_score_correction_bias", None),
        ),
        (
            "model.layers.3.block_sparse_moe.gate.e_score_correction_bias",
            (MOE + "e_score_correction_bias", None),
        ),
        # Native name: untouched.
        (MOE + "e_score_correction_bias", (MOE + "e_score_correction_bias", None)),
        (LM + "layers.3.mlp.gate.weight", (MOE + "gate.weight", None)),
        # Routed experts keep their scale tensors.
        (LM + "layers.3.mlp.experts.7.gate_proj.weight", (MOE + "experts.7.w1.weight", None)),
        (LM + "layers.3.mlp.experts.7.up_proj.weight_scale", (MOE + "experts.7.w3.weight_scale", None)),
        (LM + "layers.3.mlp.experts.7.down_proj.input_scale", (MOE + "experts.7.w2.input_scale", None)),
        (LM + "layers.3.mlp.experts.7.down_proj.weight_scale_2", (MOE + "experts.7.w2.weight_scale_2", None)),
        (
            LM + "layers.3.mlp.shared_experts.gate_up_proj.weight",
            (MOE + "shared_experts.gate_up_proj.weight", None),
        ),
        # Dense layers 0-2 keep their plain `mlp`.
        (LM + "layers.0.mlp.gate_up_proj.weight", ("model.layers.0.mlp.gate_up_proj.weight", None)),
        (LM + "layers.0.mlp.down_proj.weight", ("model.layers.0.mlp.down_proj.weight", None)),
        (LM + "layers.3.self_attn.indexer.q_proj.weight", ("model.layers.3.self_attn.qkv_proj.weight", "index_q")),
        (LM + "layers.3.self_attn.indexer.k_norm.weight", ("model.layers.3.self_attn.index_k_norm.weight", None)),
        (LM + "layers.3.self_attn.q_proj.weight", ("model.layers.3.self_attn.qkv_proj.weight", "q")),
    ],
)
def test_export_name_lands_on_the_backbone(name, expected):
    assert mapped(name) == expected


@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
@pytest.mark.parametrize("leaf", ["input_quantizer", "weight_quantizer"])
def test_modelopt_quantizer_placeholders_are_dropped(projection, leaf):
    # 87,552 one-element bf16 tensors named `experts.<proj>.<E>.<leaf>`. Left
    # alone they resolve to `layers.N.mlp…`, which a MoE layer does not have.
    assert mapped(f"{LM}layers.3.mlp.experts.{projection}.7.{leaf}") is None
    # The real per-expert tensors next to them are kept.
    assert mapped(f"{LM}layers.3.mlp.experts.7.{projection}.weight") is not None


def test_no_moe_layer_name_is_left_under_mlp():
    patterns = [
        "mlp.experts.7.{p}.{leaf}".format(p=p, leaf=leaf)
        for p in ("gate_proj", "up_proj", "down_proj")
        for leaf in ("input_scale", "weight", "weight_scale", "weight_scale_2")
    ] + [
        "mlp.experts.{p}.7.{leaf}".format(p=p, leaf=leaf)
        for p in ("gate_proj", "up_proj", "down_proj")
        for leaf in ("input_quantizer", "weight_quantizer")
    ] + [
        "mlp.gate.e_score_correction_bias",
        "mlp.gate.weight",
        "mlp.shared_experts.down_proj.weight",
        "mlp.shared_experts.gate_up_proj.weight",
    ]
    for pattern in patterns:
        result = mapped(f"{LM}layers.3.{pattern}")
        assert result is None or ".layers.3.mlp." not in result[0], (pattern, result)
