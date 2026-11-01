# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 (B12X trellis_t256) routed experts with unquantized dense layers.

The checkpoint carries routed experts in the B12X ``exl3-v1`` sidecar
container (``exl3-manifest.json`` plus one safetensors per MoE layer);
every other tensor is plain BF16/FP16 and loads unquantized.  Each rank
reads only its slot extent of every layer file, adapts it through
``b12x.moe.checkpoints.exl3.trellis_from_exl3`` and prepares it with the
planned B12X trellis API.  Adapted from local-inference-lab's
``feat/kk-kimi-exl3-experts`` loader (6b9c5d23e), retargeted to the
renamed TrellisSource API and to checkpoints whose dense sea was never
quantized (no intermediate Hadamard, equal slot extents).
"""

import re
from pathlib import Path
from typing import Any

import torch

from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.b12x import B12xExperts
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

logger = init_logger(__name__)

EXL3_MANIFEST_FILENAME = "exl3-manifest.json"
EXL3_CODEC = "EXL3_K2_LUT_E4M3"


def make_exl3_activation_spec(activation_spec: Any, nonlinearity: str) -> Any:
    """B12X activation contract for EXL3 routed experts.

    Public activations stay BF16, while the trellis boundary rotations
    execute through B12X's FP16 full-rotation path.
    """
    return activation_spec(
        mode="a16",
        nonlinearity=nonlinearity,
        io_dtype=torch.bfloat16,
        rotation_dtype=torch.float16,
    )


def plan_exl3_extent(
    num_slots: int, slot_channels: int, tp_world: int, tp_rank: int,
    intermediate_size_per_partition: int,
) -> tuple[int, int]:
    """Equal contiguous slot extent for one TP rank.

    The container reshardes at 32-channel slot granularity, so the
    per-partition intermediate must be a whole number of slots and the
    slot count must divide evenly across ranks.  Returns
    ``(first_slot, slot_count)``.
    """
    if intermediate_size_per_partition % slot_channels:
        raise ValueError(
            "EXL3 intermediate partition "
            f"{intermediate_size_per_partition} is not a multiple of the "
            f"{slot_channels}-channel slot"
        )
    slot_count = intermediate_size_per_partition // slot_channels
    if slot_count * tp_world != num_slots:
        raise ValueError(
            f"EXL3 slot count {num_slots} does not split evenly across "
            f"TP={tp_world} at {intermediate_size_per_partition} channels "
            "per partition"
        )
    return tp_rank * slot_count, slot_count


DENSE_FP8_SCHEME = "per_tensor_e4m3"
DENSE_FP8_MIN_ELEMENTS = 65536
_DENSE_FP8_EXCLUDE = ("block_sparse_moe.gate", "embed_tokens", "routed_expert_")


def dense_fp8_applies(prefix: str, input_size: int, output_size: int) -> bool:
    """True iff ``prefix`` names a dense-FP8-quantized Linear weight.

    Mirrors ``quantizable`` in tools/quantize_exl3_dense_fp8.py: the
    transform quantizes exactly the checkpoint tensors this predicate
    selects, so keep the two in lockstep.
    """
    if not prefix.startswith("language_model.model."):
        return False
    if input_size * output_size < DENSE_FP8_MIN_ELEMENTS:
        return False
    return not any(part in prefix for part in _DENSE_FP8_EXCLUDE)


class Exl3Config(QuantizationConfig):
    """EXL3 routed experts; dense layers BF16 or per-tensor FP8."""

    def __init__(
        self,
        manifest: str = EXL3_MANIFEST_FILENAME,
        dense_fp8: bool = False,
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.dense_fp8 = dense_fp8
        self._fp8_config = (
            Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
            )
            if dense_fp8
            else None
        )

    def __repr__(self) -> str:
        return (
            f"Exl3Config(manifest={self.manifest!r}, "
            f"dense_fp8={self.dense_fp8!r})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "exl3"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # The B12X trellis kernels target SM12x; the method constructor
        # enforces the real device gate.  120 keeps older devices out of
        # kernel selection entirely.
        return 120

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        exl3 = config.get("exl3") or {}
        if not isinstance(exl3, dict):
            raise ValueError("EXL3 quantization_config.exl3 must be a mapping")
        manifest = exl3.get("manifest", EXL3_MANIFEST_FILENAME)
        if manifest != EXL3_MANIFEST_FILENAME:
            raise ValueError(
                f"EXL3 manifest must be {EXL3_MANIFEST_FILENAME!r} at the "
                f"checkpoint root, got {manifest!r}"
            )
        codec = exl3.get("codec")
        if codec is not None and codec != EXL3_CODEC:
            raise ValueError(f"unsupported EXL3 codec {codec!r}")
        dense_fp8 = config.get("dense_fp8")
        if dense_fp8 is None:
            return cls(manifest=manifest)
        if not isinstance(dense_fp8, dict) or dense_fp8.get(
            "scheme"
        ) != DENSE_FP8_SCHEME:
            raise ValueError(
                "EXL3 quantization_config.dense_fp8 must be "
                f"{{'scheme': {DENSE_FP8_SCHEME!r}}}, got {dense_fp8!r}"
            )
        return cls(manifest=manifest, dense_fp8=True)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, RoutedExperts):
            return Exl3MoEMethod(layer.moe_config)
        if isinstance(layer, LinearBase):
            if self.dense_fp8 and dense_fp8_applies(
                prefix, layer.input_size, layer.output_size
            ):
                assert self._fp8_config is not None
                return Fp8LinearMethod(self._fp8_config)
            return UnquantizedLinearMethod()
        if isinstance(layer, ParallelLMHead):
            return UnquantizedLinearMethod()
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []


class Exl3MoEMethod(FusedMoEMethodBase):
    """EXL3 routed experts prepared directly from the sidecar container."""

    def __init__(self, moe_config: FusedMoEConfig) -> None:
        super().__init__(moe_config)
        if moe_config.moe_backend not in ("auto", "b12x"):
            raise ValueError("EXL3 routed experts require the b12x MoE backend")
        if not B12xExperts._supports_current_device():
            raise ValueError("EXL3 routed experts require b12x on SM12x")
        if not B12xExperts._supports_parallel_config(
            moe_config.moe_parallel_config
        ):
            raise ValueError(
                "EXL3 supports tensor parallelism without EP or EPLB"
            )
        if moe_config.in_dtype != torch.bfloat16:
            raise ValueError("EXL3 routed experts require BF16 activations")
        if (
            moe_config.activation not in (MoEActivation.SILU, MoEActivation.SITU)
            or moe_config.has_bias
        ):
            raise ValueError(
                "EXL3 routed experts require bias-free SiLU or SiTU experts"
            )
        if moe_config.activation == MoEActivation.SITU and (
            moe_config.activation_situ_beta != 4.0
            or moe_config.activation_situ_linear_beta != 25.0
        ):
            raise ValueError(
                "B12X EXL3 MoE supports only SiTU beta 4.0 / linear beta 25.0"
            )
        if any(
            value is not None
            for value in (
                moe_config.swiglu_limit,
                moe_config.swiglu_alpha,
                moe_config.swiglu_beta,
            )
        ):
            raise ValueError("EXL3 routed experts require standard SiLU parameters")

    def maybe_roundup_sizes(
        self,
        hidden_size,
        intermediate_size_per_partition,
        act_dtype,
        moe_parallel_config,
    ):
        if hidden_size % 128:
            raise ValueError(
                f"EXL3 hidden size must align to 128, got {hidden_size}"
            )
        if intermediate_size_per_partition % 32:
            raise ValueError(
                "EXL3 intermediate partition must align to the 32-channel "
                f"slot, got {intermediate_size_per_partition}"
            )
        return hidden_size, intermediate_size_per_partition

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype != torch.bfloat16:
            raise ValueError("EXL3 experts require BF16 activations")
        self.maybe_roundup_sizes(
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            self.moe.moe_parallel_config,
        )
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", layer.layer_name)
        if match is None:
            raise ValueError("EXL3 experts require a numbered MoE layer")
        self.layer_index = int(match.group(1))
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size_per_partition
        # No checkpoint tensors exist for the routed experts: the modular
        # kernel resolves geometry and storage through the prepared B12X
        # package installed in process_weights_after_loading.
        for name in ("w13_weight", "w2_weight"):
            layer.register_parameter(
                name,
                torch.nn.Parameter(
                    torch.empty(0, dtype=torch.uint8), requires_grad=False
                ),
            )

    def get_fused_moe_quant_config(self, layer):
        return FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(),
            _a2=FusedMoEQuantDesc(),
            _w1=FusedMoEQuantDesc(dtype="exl3"),
            _w2=FusedMoEQuantDesc(dtype="exl3"),
        )

    def process_weights_after_loading(self, layer) -> None:
        from b12x.moe import fused_moe
        from b12x.moe.checkpoints.exl3 import (
            read_exl3_layer,
            read_exl3_manifest,
            trellis_from_exl3,
        )

        config = get_current_vllm_config()
        root = Path(config.model_config.model)
        manifest = read_exl3_manifest(root)
        geometry = manifest.geometry
        if (geometry.num_experts, geometry.hidden_size) != (
            self.num_experts,
            self.hidden_size,
        ):
            raise ValueError(
                "EXL3 manifest geometry does not match the routed-expert layer"
            )
        if manifest.hadamard.intermediate_hadamard:
            raise ValueError(
                "this loader serves ordinary-boundary EXL3 checkpoints only "
                "(intermediate_hadamard=false)"
            )
        tp = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        first_slot, slot_count = plan_exl3_extent(
            geometry.num_slots,
            geometry.slot_channels,
            tp,
            rank,
            self.intermediate_size_per_partition,
        )
        source_data = read_exl3_layer(
            root,
            manifest,
            self.layer_index,
            first_slot=first_slot,
            slot_count=slot_count,
        )
        source, weights = trellis_from_exl3(source_data)
        device = layer.w13_weight.device
        plan = fused_moe.plan_weights(
            source=source,
            activation=make_exl3_activation_spec(
                fused_moe.ActivationSpec,
                "situ" if layer.activation == MoEActivation.SITU else "silu",
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=geometry.num_experts,
                hidden_size=geometry.hidden_size,
                intermediate_size=source_data.local_intermediate_size,
            ),
        )
        prepared = fused_moe.prepare_weights(
            plan=plan, weights=weights, device=device
        )
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        prepare_finalize = maybe_make_prepare_finalize(
            moe=self.moe,
            quant_config=self.moe_quant_config,
            routing_tables=layer._expert_routing_tables(),
            allow_new_interface=True,
        )
        assert prepare_finalize is not None
        experts = B12xExperts(self.moe, self.moe_quant_config)
        experts.install_prepared_experts(layer, prepared)
        self.moe_kernel = mk.FusedMoEKernel(prepare_finalize, experts)
        logger.info(
            "EXL3 layer %d rank %d/%d: slots [%d,%d), %d channels, "
            "%s %d-bit ordinary boundary; BF16 activations",
            self.layer_index,
            rank,
            tp,
            first_slot,
            first_slot + slot_count,
            source_data.local_intermediate_size,
            manifest.codebook,
            manifest.rates.bits,
        )

    def apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts,
        shared_experts_input,
    ):
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )


__all__ = [
    "Exl3Config",
    "Exl3MoEMethod",
    "dense_fp8_applies",
    "plan_exl3_extent",
]
