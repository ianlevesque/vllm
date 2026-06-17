# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    TRITON_BACKENDS,
    Mxfp4MoeBackend,
    convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    convert_weight_to_mxfp4_moe_kernel_format,
    make_mxfp4_moe_kernel,
    make_mxfp4_moe_quant_config,
    mxfp4_round_up_hidden_size_and_intermediate_size,
    select_deepseek_v4_mxfp4_moe_backend,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform

logger = init_logger(__name__)


class Mxfp4Config(QuantizationConfig):
    """Canonical base config for MXFP4 quantization.

    Subclasses override get_name() and override_quantization_method() to
    register themselves as the handler for a specific checkpoint format.
    """

    def __init__(self, ignored_layers: list[str] | None = None):
        super().__init__()
        self.ignored_layers = ignored_layers

    @classmethod
    def from_config(cls, config):
        return cls()

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    # TODO (zyongye) This is only temporaty fallback.
    # We should have `Mxfp4MoEMethod` after this migration is complete.
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            if self.ignored_layers and is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            logger.debug_once(
                "MXFP4 linear layer is not implemented - falling back to "
                "UnquantizedLinearMethod.",
            )
            return UnquantizedLinearMethod()
        elif isinstance(layer, RoutedExperts):
            return GptOssMxfp4MoEMethod(layer.moe_config)
        elif isinstance(layer, Attention):
            logger.debug_once(
                "MXFP4 attention layer is not implemented. "
                "Skipping quantization for this layer.",
            )
        return None

    def is_mxfp4_quant(self, prefix: str, layer: torch.nn.Module) -> bool:
        """MXFP4 config always uses MXFP4 quantization."""
        return True


class GptOssMxfp4Config(Mxfp4Config):
    """MXFP4 config for GPT-OSS checkpoints.

    Checkpoints carry ``"quant_method": "mxfp4"`` in their JSON config.
    override_quantization_method() maps that to the canonical internal name
    so that the rest of the loading path uses "gpt_oss_mxfp4" consistently.
    """

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "gpt_oss_mxfp4"

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        # Match both "mxfp4" (original checkpoint value) and "gpt_oss_mxfp4"
        # (already normalized by verify_and_update_model_config) so that
        # explicit --quantization mxfp4 from the user doesn't cause a mismatch.
        if not (
            isinstance(hf_quant_cfg, dict)
            and hf_quant_cfg.get("quant_method") in ("mxfp4", "gpt_oss_mxfp4")
        ):
            return None
        # Require explicit confirmation that this is a GPT-OSS model.
        # Do NOT fall back to returning the override when hf_config is None,
        # as that would silently claim all mxfp4 checkpoints.
        model_type = getattr(hf_config, "model_type", None)
        if model_type != "gpt_oss":
            return None
        return "gpt_oss_mxfp4"


class GptOssMxfp4MoEMethod(FusedMoEMethodBase):
    """MXFP4 MoE quantization method."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.weight_dtype = "gpt_oss_mxfp4"
        self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(moe)

        self.max_capture_size = moe.max_capture_size

        self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
        self.moe_kernel: mk.FusedMoEKernel | None = None

        # Used for triton kernel precision configs
        self.w13_precision_config = None
        self.w2_precision_config = None

    @property
    def skip_forward_padding(self) -> bool:
        # SM100_FI_MXFP4_MXFP8_TRTLLM supports padding with mxfp8 quant
        # so can skip the padding in the forward before applying the moe method
        return self.mxfp4_backend == Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8

    # TODO(bnell): move to MK/expert_class?
    @property
    def has_unpadded_output(self) -> bool:
        return self.mxfp4_backend in [
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
        ]

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        hidden_size, intermediate_size_per_partition = super().maybe_roundup_sizes(
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            act_dtype=act_dtype,
            moe_parallel_config=moe_parallel_config,
        )
        return mxfp4_round_up_hidden_size_and_intermediate_size(
            self.mxfp4_backend, hidden_size, intermediate_size_per_partition
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.num_experts = num_experts
        weight_dtype = torch.uint8
        scale_dtype = torch.uint8
        mxfp4_block = 32

        layer.params_dtype = params_dtype
        layer.num_experts = num_experts
        self.intermediate_size = intermediate_size_per_partition
        self.hidden_size = hidden_size

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        w13_weight_scale.quant_method = "block"

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        w2_weight_scale.quant_method = "block"

        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    hidden_size,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def _setup_kernel(
        self,
        layer: RoutedExperts,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
    ) -> None:
        num_experts = self.num_experts
        intermediate_size = self.intermediate_size
        hidden_size = self.hidden_size
        sf_block_size = 32

        # Shape assertions
        assert (
            w13.dim() == 3
            and w13.shape[0] == num_experts
            and w13.shape[1] == intermediate_size * 2
            and w13.shape[2] == hidden_size // 2
        )
        assert (
            w13_scale.dim() == 3
            and w13_scale.shape[0] == num_experts
            and w13_scale.shape[1] == intermediate_size * 2
            and w13_scale.shape[2] == hidden_size // sf_block_size
        )
        assert (
            w2.dim() == 3
            and w2.shape[0] == num_experts
            and w2.shape[1] == hidden_size
            and w2.shape[2] == intermediate_size // 2
        )
        assert (
            w2_scale.dim() == 3
            and w2_scale.shape[1] == hidden_size
            and w2_scale.shape[2] == intermediate_size // sf_block_size
        )
        if w13_bias is not None:
            assert (
                w13_bias.dim() == 2
                and w13_bias.shape[0] == num_experts
                and w13_bias.shape[1] == intermediate_size * 2
            )
        if w2_bias is not None:
            assert (
                w2_bias.dim() == 2
                and w2_bias.shape[0] == num_experts
                and w2_bias.shape[1] == hidden_size
            )

        # Convert weights to kernel format
        w13, w2, w13_scale, w2_scale, w13_bias, w2_bias = (
            convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
                mxfp4_backend=self.mxfp4_backend,
                layer=layer,
                w13_weight=w13,
                w2_weight=w2,
                w13_weight_scale=w13_scale,
                w2_weight_scale=w2_scale,
                w13_bias=w13_bias,
                w2_bias=w2_bias,
                _cache_permute_indices=self._cache_permute_indices,
            )
        )

        # For TRITON backends, weights are wrapped tensors from triton_kernels
        # that don't support .detach(). Manually assign parameters.
        if self.mxfp4_backend not in TRITON_BACKENDS:
            replace_parameter(layer, "w13_weight", w13)
            replace_parameter(layer, "w2_weight", w2)
            replace_parameter(layer, "w13_weight_scale", w13_scale)
            replace_parameter(layer, "w2_weight_scale", w2_scale)
        else:
            layer.w13_weight = w13
            layer.w2_weight = w2
            self.w13_precision_config = w13_scale
            self.w2_precision_config = w2_scale

        # AITER backend requires weights to be marked as shuffled.
        if self.mxfp4_backend == Mxfp4MoeBackend.AITER_MXFP4_BF16:
            layer.w13_weight.is_shuffled = True
            layer.w2_weight.is_shuffled = True

        if w13_bias is not None and w2_bias is not None:
            replace_parameter(layer, "w13_bias", w13_bias)
            replace_parameter(layer, "w2_bias", w2_bias)

        # Build quant config
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

        # Build kernel (modular or monolithic)
        if self.moe_quant_config is not None and self.experts_cls is not None:
            self.moe_kernel = make_mxfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                mxfp4_backend=self.mxfp4_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
                layer=layer,
            )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w13_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)

        if self.mxfp4_backend == Mxfp4MoeBackend.NONE:
            return

        self._setup_kernel(layer, w13, w2, w13_scale, w2_scale, w13_bias, w2_bias)

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        w1_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w1_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)

        if self.mxfp4_backend in TRITON_BACKENDS:
            assert self.w13_precision_config is not None
            assert self.w2_precision_config is not None
            w1_scale = self.w13_precision_config
            w2_scale = self.w2_precision_config

        return make_mxfp4_moe_quant_config(
            mxfp4_backend=self.mxfp4_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            gemm1_alpha=1.702,
            gemm1_beta=1.0,
            swiglu_limit=7.0,
            layer=layer,
        )

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalize,
        layer: RoutedExperts,
    ) -> mk.FusedMoEExpertsModular:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel "
            "initialization logic. This function should not be called."
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            router_logits=router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
        )


class _MiMoSm12xUnfusedOAITritonExperts:
    """sm12x (GB10 / RTX Pro 6000) gate-override for the unfused OAI Triton MoE.

    The OAI triton MoE expert classes gate ``_supports_current_device`` to the
    CUDA window ``(9,0) <= cap < (11,0)`` (Hopper SM90 + Blackwell SM100),
    explicitly excluding SM120. The triton_kernels matmul_ogs + swiglu kernels
    are SM-agnostic and run correctly on SM121, so for the MiMo case we widen
    that gate to SM12x. Everything else (W4A16 mxfp4 weight scheme, SILU
    activation via ``swiglu_limit_func`` = silu(gate)*up, routing) is inherited
    unchanged from ``UnfusedOAITritonExperts``. Kept MiMo-local so gpt-oss and
    the generic Mxfp4MoEMethod are unaffected.
    """

    @staticmethod
    def _supports_current_device() -> bool:
        from vllm.utils.import_utils import has_triton_kernels

        return (
            current_platform.is_device_capability_family(120) and has_triton_kernels()
        )


class MiMoV2Mxfp4MoEMethod(GptOssMxfp4MoEMethod):
    """MiMo MXFP4 MoE packing compatible with its DFlash checkpoint."""

    def __init__(self, moe: FusedMoEConfig, swiglu_limit: float | None = None):
        FusedMoEMethodBase.__init__(self, moe)
        self.weight_dtype = "mxfp4"
        if not hasattr(moe, "max_capture_size"):
            moe.max_capture_size = 0
        self.mxfp4_backend, self.experts_cls = select_deepseek_v4_mxfp4_moe_backend(moe)
        self.max_capture_size = moe.max_capture_size
        self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
        self.moe_kernel: mk.FusedMoEKernel | None = None
        self.w13_precision_config = None
        self.w2_precision_config = None
        self._swiglu_limit = swiglu_limit

        # SM12x (GB10 DGX Spark / RTX Pro 6000): the sm120 cutlass MXFP4xMXFP8
        # fused-MoE kernel does not consume the block scales correctly (garbled
        # output), and there is no sm120 trtllm-gen MoE cubin. Route MiMo's
        # routed-experts MoE through the SM-agnostic, gpt-oss-validated TRITON
        # path instead: dequant-free mxfp4 weights matmul'd against *bf16*
        # activations (W4A16 — drop the mxfp8-act spec; bf16 act is numerically
        # correct, just unquantized). We use the UNFUSED OAI triton experts
        # (Mxfp4MoeBackend.TRITON_UNFUSED) rather than the fused OAITritonExperts
        # because the fused matmul_ogs swiglu hardcodes gpt-oss params
        # (alpha=1.702, clamp=7.0) and computes silu(gate)*(up+1) (gemm1_beta=1),
        # which garbles MiMo's STANDARD SwiGLU. The unfused class lists SILU in
        # _supports_activation and runs silu(gate)*up via swiglu_limit_func
        # (no +1), matching MiMo's config hidden_act=silu. Both triton backends
        # share the same _swizzle_mxfp4 + PrecisionConfig weight conversion and
        # the mxfp4_w4a16 quant config, so the inherited gpt-oss triton
        # _setup_kernel path is reused verbatim (see process_weights_after_loading).
        # Off sm12x, keep the original TRT-LLM modular expert path.
        self._use_sm12x_triton_moe = current_platform.is_device_capability_family(120)
        if self._use_sm12x_triton_moe:
            import os

            from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (  # noqa: E501
                UnfusedOAITritonExperts,
                _MiMoSm12xFusedOAITritonExperts,
            )

            self._mimo_fused_moe = os.environ.get("MIMO_FUSED_MOE", "0") == "1"
            if self._mimo_fused_moe:
                # FUSED OAI TRITON path (#19 perf): folds silu_and_mul into matmul1
                # via the beta=0 fused swiglu epilogue (no "+1"; alpha=1, no clamp).
                # Faster than the unfused path (no separate activation kernel +
                # intermediate r/w). Same _swizzle_mxfp4 / mxfp4_w4a16 weight prep,
                # so process_weights_after_loading is reused verbatim. Gated behind
                # the env toggle so the SAME image A/Bs vs the proven unfused path
                # and falls back instantly if it regresses.
                self.mxfp4_backend = Mxfp4MoeBackend.TRITON
                self.experts_cls = _MiMoSm12xFusedOAITritonExperts
                logger.info_once(
                    "MiMo MXFP4 MoE: FUSED OAI TRITON path (beta=0 swiglu, "
                    "Mxfp4MoeBackend.TRITON) on SM12x [MIMO_FUSED_MOE=1]."
                )
            else:
                # MiMo-local subclass of UnfusedOAITritonExperts that widens the
                # device gate to SM12x (the only change). backend_to_kernel_cls is
                # not used directly because the stock device gate would reject the
                # class on SM120 before selection.
                class _MiMoUnfusedOAITritonExperts(
                    _MiMoSm12xUnfusedOAITritonExperts, UnfusedOAITritonExperts
                ):
                    pass

                self.mxfp4_backend = Mxfp4MoeBackend.TRITON_UNFUSED
                self.experts_cls = _MiMoUnfusedOAITritonExperts
                logger.info_once(
                    "MiMo MXFP4 MoE: routing to the unfused OAI TRITON W4A16 path "
                    "(Mxfp4MoeBackend.TRITON_UNFUSED, bf16 activations) on SM12x."
                )
        else:
            # MiMo must use vLLM's externally computed grouped sigmoid routing.
            # The monolithic TRT-LLM expert path re-routes internally and drops
            # the MiMo correction-bias routing semantics.
            from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
                TrtLlmMxfp4ExpertsModular,
            )

            self.experts_cls = TrtLlmMxfp4ExpertsModular

    @property
    def supports_eplb(self) -> bool:
        return True

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if getattr(self, "_use_sm12x_triton_moe", False):
            # SM12x TRITON W4A16 path: reuse the inherited gpt-oss triton
            # _setup_kernel verbatim. It runs the TRITON branch of
            # convert_gpt_oss_weight_to_mxfp4_moe_kernel_format (_swizzle_mxfp4 +
            # PrecisionConfig into self.w13/w2_precision_config), builds the
            # W4A16 quant config (mxfp4_w4a16_moe_quant_config — bf16 act, no
            # mxfp8) and the modular kernel. MiMo stores w13 as contiguous
            # [gate; up], which is exactly what the unfused swiglu split
            # (gate = first half, up = second half) expects — no extra reorder.
            w13_bias = getattr(layer, "w13_bias", None)
            w2_bias = getattr(layer, "w2_bias", None)
            if getattr(self, "_mimo_fused_moe", False):
                # FUSED path: the fused matmul_ogs swiglu splits gate/up INTERLEAVED
                # per output-tile (a[...,::2]=gate -> silu, a[...,1::2]=up), but MiMo's
                # w13 is contiguous [gate(0..I-1); up(0..I-1)]. Permute w13's output
                # rows to interleaved [g0,u0,g1,u1,...] so matmul1 feeds the fused
                # epilogue correctly. (The unfused path keeps contiguous — its separate
                # silu_and_mul expects [gate; up] halves.) MXFP4 packs along K, so this
                # output-dim (dim-1) row reorder is a plain index permute; the per-row
                # block scales and w13_bias reorder identically.
                _half = layer.w13_weight.shape[1] // 2
                _perm = (
                    torch.stack(
                        [torch.arange(_half), torch.arange(_half) + _half], dim=1
                    )
                    .reshape(-1)
                    .to(layer.w13_weight.device)
                )
                layer.w13_weight.data = layer.w13_weight.data[:, _perm, :].contiguous()
                layer.w13_weight_scale.data = layer.w13_weight_scale.data[
                    :, _perm, :
                ].contiguous()
                if w13_bias is not None:
                    w13_bias = w13_bias[:, _perm].contiguous()
                logger.info_once(
                    "MiMo MXFP4 MoE: interleaved w13 gate/up rows [g0,u0,...] for "
                    "the fused swiglu epilogue."
                )
            GptOssMxfp4MoEMethod._setup_kernel(
                self,
                layer,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                w13_bias,
                w2_bias,
            )
            return

        from flashinfer.fp4_quantization import block_scale_interleave
        from flashinfer.fused_moe.core import (
            _maybe_get_cached_w3_w1_permute_indices,
            get_w2_permute_indices_with_cache,
        )

        from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
            reorder_w1w3_to_w3w1,
        )

        w13_weight, w13_scale = reorder_w1w3_to_w3w1(
            layer.w13_weight.data, layer.w13_weight_scale.data, dim=-2
        )
        w2_weight = layer.w2_weight.data
        w2_scale = layer.w2_weight_scale.data

        num_experts = w13_weight.shape[0]
        epilogue_tile_m = 128
        permute_cache: dict = {}
        shuffled_w13 = []
        shuffled_w13_scale = []
        shuffled_w2 = []
        shuffled_w2_scale = []

        for expert_idx in range(num_experts):
            w13_u8 = w13_weight[expert_idx].view(torch.uint8)
            w13_scale_u8 = w13_scale[expert_idx].view(torch.uint8)
            w2_u8 = w2_weight[expert_idx].view(torch.uint8)
            w2_scale_u8 = w2_scale[expert_idx].view(torch.uint8)

            w13_perm = _maybe_get_cached_w3_w1_permute_indices(
                permute_cache, w13_u8, epilogue_tile_m
            )
            shuffled_w13.append(w13_u8[w13_perm.to(w13_u8.device)].contiguous())
            w13_scale_perm = _maybe_get_cached_w3_w1_permute_indices(
                permute_cache,
                w13_scale_u8,
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            shuffled_w13_scale.append(
                block_scale_interleave(
                    w13_scale_u8[w13_scale_perm.to(w13_scale_u8.device)].contiguous()
                )
            )

            w2_perm = get_w2_permute_indices_with_cache(
                permute_cache, w2_u8, epilogue_tile_m
            )
            shuffled_w2.append(w2_u8[w2_perm.to(w2_u8.device)].contiguous())
            w2_scale_perm = get_w2_permute_indices_with_cache(
                permute_cache,
                w2_scale_u8,
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            shuffled_w2_scale.append(
                block_scale_interleave(
                    w2_scale_u8[w2_scale_perm.to(w2_scale_u8.device)].contiguous()
                )
            )

        replace_parameter(layer, "w13_weight", torch.stack(shuffled_w13))
        replace_parameter(layer, "w2_weight", torch.stack(shuffled_w2))
        replace_parameter(
            layer,
            "w13_weight_scale",
            torch.stack(shuffled_w13_scale)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, w13_weight.shape[1], -1),
        )
        replace_parameter(
            layer,
            "w2_weight_scale",
            torch.stack(shuffled_w2_scale)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, w2_weight.shape[1], -1),
        )

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config is not None and self.experts_cls is not None:
            self.moe_kernel = make_mxfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                mxfp4_backend=self.mxfp4_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
                layer=layer,
            )

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        w1_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w1_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)

        if self.mxfp4_backend in TRITON_BACKENDS:
            assert self.w13_precision_config is not None
            assert self.w2_precision_config is not None
            w1_scale = self.w13_precision_config
            w2_scale = self.w2_precision_config

        return make_mxfp4_moe_quant_config(
            mxfp4_backend=self.mxfp4_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            swiglu_limit=self._swiglu_limit,
            layer=layer,
        )


class MiMoV2Mxfp4Config(Mxfp4Config):
    """MXFP4 config for MiMo-V2 experts."""

    SWIGLU_LIMIT = 10.0

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        # NOTE: the fused-MoE refactor replaced the `FusedMoE` class with a
        # `FusedMoE(...)` factory returning a `MoERunner`; the routed-expert
        # module that reaches quant-method selection is a `RoutedExperts`
        # (matching the parent `Mxfp4Config.get_quant_method`). Checking the
        # old `FusedMoE` symbol (now a function) would raise TypeError.
        if isinstance(layer, RoutedExperts):
            return MiMoV2Mxfp4MoEMethod(
                layer.moe_config, swiglu_limit=self.SWIGLU_LIMIT
            )
        return super().get_quant_method(layer, prefix)


class Mxfp4MoEMethod(FusedMoEMethodBase):
    """MXFP4 MoE quantization method."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.weight_dtype = "mxfp4"
        self.mxfp4_backend, self.experts_cls = select_deepseek_v4_mxfp4_moe_backend(moe)

        self.max_capture_size = moe.max_capture_size

        self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
        self.moe_kernel: mk.FusedMoEKernel | None = None

        # Used for triton kernel precision configs
        self.w13_precision_config = None
        self.w2_precision_config = None

    @property
    def supports_eplb(self) -> bool:
        return True

    @property
    def skip_forward_padding(self) -> bool:
        # SM100_FI_MXFP4_MXFP8_TRTLLM supports padding with mxfp8 quant
        # so can skip the padding in the forward before applying the moe method
        return self.mxfp4_backend == Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8

    # TODO(bnell): move to MK/expert_class?
    @property
    def has_unpadded_output(self) -> bool:
        return self.mxfp4_backend in [
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
        ]

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        hidden_size, intermediate_size_per_partition = super().maybe_roundup_sizes(
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            act_dtype=act_dtype,
            moe_parallel_config=moe_parallel_config,
        )
        return mxfp4_round_up_hidden_size_and_intermediate_size(
            self.mxfp4_backend, hidden_size, intermediate_size_per_partition
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.num_experts = num_experts
        weight_dtype = torch.uint8
        scale_dtype = torch.uint8
        mxfp4_block = 32

        layer.params_dtype = params_dtype
        layer.num_experts = num_experts
        self.intermediate_size = intermediate_size_per_partition
        self.hidden_size = hidden_size

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        w13_weight_scale.quant_method = "block"

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        w2_weight_scale.quant_method = "block"

        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    hidden_size,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def _setup_kernel(
        self,
        layer: RoutedExperts,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
    ) -> None:
        num_experts = self.num_experts
        intermediate_size = self.intermediate_size
        hidden_size = self.hidden_size
        sf_block_size = 32

        # Shape assertions
        assert (
            w13.dim() == 3
            and w13.shape[0] == num_experts
            and w13.shape[1] == intermediate_size * 2
            and w13.shape[2] == hidden_size // 2
        )
        assert (
            w13_scale.dim() == 3
            and w13_scale.shape[0] == num_experts
            and w13_scale.shape[1] == intermediate_size * 2
            and w13_scale.shape[2] == hidden_size // sf_block_size
        )
        assert (
            w2.dim() == 3
            and w2.shape[0] == num_experts
            and w2.shape[1] == hidden_size
            and w2.shape[2] == intermediate_size // 2
        )
        assert (
            w2_scale.dim() == 3
            and w2_scale.shape[1] == hidden_size
            and w2_scale.shape[2] == intermediate_size // sf_block_size
        )
        if w13_bias is not None:
            assert (
                w13_bias.dim() == 2
                and w13_bias.shape[0] == num_experts
                and w13_bias.shape[1] == intermediate_size * 2
            )
        if w2_bias is not None:
            assert (
                w2_bias.dim() == 2
                and w2_bias.shape[0] == num_experts
                and w2_bias.shape[1] == hidden_size
            )

        # Convert weights to kernel format
        w13, w2, w13_scale, w2_scale, w13_bias, w2_bias = (
            convert_weight_to_mxfp4_moe_kernel_format(
                mxfp4_backend=self.mxfp4_backend,
                layer=layer,
                w13_weight=w13,
                w2_weight=w2,
                w13_weight_scale=w13_scale,
                w2_weight_scale=w2_scale,
                w13_bias=w13_bias,
                w2_bias=w2_bias,
                _cache_permute_indices=self._cache_permute_indices,
            )
        )

        # For TRITON backends, weights are wrapped tensors from triton_kernels
        # that don't support .detach(). Manually assign parameters.
        if self.mxfp4_backend not in TRITON_BACKENDS:
            replace_parameter(layer, "w13_weight", w13)
            replace_parameter(layer, "w2_weight", w2)
            replace_parameter(layer, "w13_weight_scale", w13_scale)
            replace_parameter(layer, "w2_weight_scale", w2_scale)
        else:
            layer.w13_weight = w13
            layer.w2_weight = w2
            self.w13_precision_config = w13_scale
            self.w2_precision_config = w2_scale

        # AITER backend requires weights to be marked as shuffled.
        if self.mxfp4_backend == Mxfp4MoeBackend.AITER_MXFP4_BF16:
            layer.w13_weight.is_shuffled = True
            layer.w2_weight.is_shuffled = True

        if w13_bias is not None and w2_bias is not None:
            replace_parameter(layer, "w13_bias", w13_bias)
            replace_parameter(layer, "w2_bias", w2_bias)

        # Build quant config
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

        # Build kernel (modular or monolithic)
        if self.moe_quant_config is not None and self.experts_cls is not None:
            self.moe_kernel = make_mxfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                mxfp4_backend=self.mxfp4_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
                layer=layer,
            )

    def process_weights_after_loading(self, layer):
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w13_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)

        if self.mxfp4_backend == Mxfp4MoeBackend.NONE:
            return

        self._setup_kernel(layer, w13, w2, w13_scale, w2_scale, w13_bias, w2_bias)

    def get_fused_moe_quant_config(
        self,
        layer: RoutedExperts,
    ) -> FusedMoEQuantConfig | None:
        w1_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w1_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)
        swiglu_limit = getattr(layer, "swiglu_limit", None)

        if self.mxfp4_backend in TRITON_BACKENDS:
            assert self.w13_precision_config is not None
            assert self.w2_precision_config is not None
            w1_scale = self.w13_precision_config
            w2_scale = self.w2_precision_config

        return make_mxfp4_moe_quant_config(
            mxfp4_backend=self.mxfp4_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            swiglu_limit=swiglu_limit,
            layer=layer,
        )

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalize,
        layer: RoutedExperts,
    ) -> mk.FusedMoEExpertsModular:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel "
            "initialization logic. This function should not be called."
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            router_logits=router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
        )
