# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 decode GEMM selection for unquantized BF16 on SM103 and SM121.

Dispatch is purely by local ``(N, K)`` shape and token count ``M`` — the module
name plays no role. Each measured shape maps to a :class:`ProjectionSpec`
holding the winning backend per token count. The static part of the decision is
resolved once per module at install time into a small ``{M: call}`` plan, so the
per-forward path is a single dict lookup.

SM103 uses the full projection table. SM121 (GB10) routes only
kernel-validate-measured >=1.05x winners: the DSV3 map below is authoritative
(several SM103 winners lose against cuBLAS there, and K=7168 shapes were
unmeasurable before the smem fix), while CuTe plans are filtered to the
winner subset. Anything not listed stays on the default GEMM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform

Backend = Literal["cute", "dsv3_fused_a"]
# A resolved per-token-count call: the backend plus its CuTe config (None for
# dsv3, which needs no config).
ResolvedCall = tuple[Backend, SkinnyGemmConfig | None]


@dataclass(frozen=True, slots=True)
class ProjectionSpec:
    n: int
    k: int
    dsv3_tokens: frozenset[int] = frozenset()
    cute_configs: tuple[tuple[int, SkinnyGemmConfig], ...] = ()
    residual_configs: tuple[tuple[int, SkinnyGemmConfig], ...] = ()
    name: str = ""  # optional debug label; never used for dispatch

    def cute_config(self, num_tokens: int) -> SkinnyGemmConfig | None:
        return dict(self.cute_configs).get(num_tokens)

    def residual_config(self, num_tokens: int) -> SkinnyGemmConfig | None:
        return dict(self.residual_configs).get(num_tokens)


def _cute(
    num_tokens: int,
    block_size: int,
    outputs_per_block: int,
    k_unroll: int,
    vector_width: int = 8,
) -> SkinnyGemmConfig:
    return SkinnyGemmConfig(
        num_tokens,
        block_size,
        outputs_per_block,
        k_unroll,
        vector_width,
    )


_M1_TO_16 = frozenset(range(1, 17))
_M1 = frozenset({1})

# Keyed by local (N, K). Where two projections share a shape (only 1536x7168:
# shared_gate_up_proj and mla_g_proj) the entry is unified.
KIMI_K3_PROJECTIONS: dict[tuple[int, int], ProjectionSpec] = {
    (1536, 128): ProjectionSpec(1536, 128, _M1_TO_16, name="f_b_proj"),
    (3072, 128): ProjectionSpec(3072, 128, _M1_TO_16, name="f_b_proj"),
    # 1536x7168 is shared by shared_gate_up_proj and mla_g_proj. dsv3 M1..16 is
    # only crash-safe once the mla_g aux-stream/PDL capture fix lands (subtask
    # task_7388aba1); the fallback if it cannot be fixed is dsv3_tokens=_M1.
    (1536, 7168): ProjectionSpec(
        1536, 7168, _M1_TO_16, name="shared_gate_up_proj/mla_g_proj"
    ),
    (3072, 7168): ProjectionSpec(
        3072,
        7168,
        cute_configs=(
            (1, _cute(1, 224, 3, 4)),
            (2, _cute(2, 128, 3, 2)),
            (3, _cute(3, 128, 2, 1)),
            (4, _cute(4, 64, 2, 2)),
            (5, _cute(5, 128, 3, 1)),
        ),
        name="shared_gate_up_proj",
    ),
    (2112, 7168): ProjectionSpec(2112, 7168, _M1_TO_16, name="fused_qkv_a_proj"),
    (2304, 1536): ProjectionSpec(2304, 1536, _M1_TO_16, name="q_b_proj"),
    (4608, 1536): ProjectionSpec(4608, 1536, _M1_TO_16, name="q_b_proj"),
    (3584, 7168): ProjectionSpec(
        3584,
        7168,
        frozenset(range(2, 9)),
        ((1, _cute(1, 224, 2, 4)),),
        name="routed_expert_down_proj",
    ),
    (6288, 7168): ProjectionSpec(
        6288,
        7168,
        cute_configs=(
            (1, _cute(1, 224, 3, 4)),
            (2, _cute(2, 64, 3, 2)),
            (3, _cute(3, 32, 3, 4)),
            (4, _cute(4, 128, 6, 1)),
        ),
        name="in_proj_qkvgfab",
    ),
    (12448, 7168): ProjectionSpec(
        12448,
        7168,
        cute_configs=(
            (1, _cute(1, 224, 4, 2)),
            (2, _cute(2, 64, 4, 2)),
            (3, _cute(3, 64, 2, 2)),
        ),
        name="in_proj_qkvgfab",
    ),
    (7168, 768): ProjectionSpec(7168, 768, _M1_TO_16, name="shared_down_proj"),
    (7168, 1536): ProjectionSpec(
        7168, 1536, cute_configs=((1, _cute(1, 96, 4, 2)),), name="o_proj"
    ),
    (7168, 3072): ProjectionSpec(
        7168,
        3072,
        cute_configs=(
            (1, _cute(1, 96, 2, 4)),
            (2, _cute(2, 32, 4, 4)),
        ),
        name="o_proj",
    ),
    (7168, 3584): ProjectionSpec(
        7168,
        3584,
        cute_configs=(
            (1, _cute(1, 224, 4, 2)),
            (2, _cute(2, 64, 4, 2)),
        ),
        residual_configs=(
            (1, _cute(1, 64, 4, 2)),
            (2, _cute(2, 64, 7, 2)),
            (3, _cute(3, 64, 2, 1)),
            (4, _cute(4, 64, 2, 1)),
        ),
        name="routed_expert_up_proj",
    ),
    (7168, 4224): ProjectionSpec(
        7168,
        4224,
        cute_configs=((1, _cute(1, 96, 4, 2, 4)),),
        name="dense_down_proj",
    ),
    (7168, 8448): ProjectionSpec(
        7168,
        8448,
        cute_configs=(
            (1, _cute(1, 32, 4, 4)),
            (2, _cute(2, 96, 4, 1)),
            (3, _cute(3, 96, 4, 1)),
        ),
        name="dense_down_proj",
    ),
    (8448, 7168): ProjectionSpec(
        8448,
        7168,
        cute_configs=(
            (1, _cute(1, 224, 3, 4)),
            (2, _cute(2, 32, 4, 4)),
        ),
        name="dense_gate_up_proj",
    ),
    (16896, 7168): ProjectionSpec(
        16896,
        7168,
        cute_configs=(
            (1, _cute(1, 224, 6, 4)),
            (2, _cute(2, 32, 4, 4)),
        ),
        name="dense_gate_up_proj",
    ),
    (20480, 7168): ProjectionSpec(
        20480,
        7168,
        cute_configs=(
            (1, _cute(1, 224, 4, 2)),
            (2, _cute(2, 64, 4, 2)),
            (3, _cute(3, 64, 2, 2)),
            (4, _cute(4, 64, 4, 1)),
        ),
        name="lm_head",
    ),
    (40960, 7168): ProjectionSpec(
        40960,
        7168,
        cute_configs=(
            (1, _cute(1, 128, 4, 2)),
            (2, _cute(2, 64, 4, 2)),
            (3, _cute(3, 64, 2, 2)),
            (4, _cute(4, 64, 4, 1)),
        ),
        name="lm_head",
    ),
    # TP16. Measured on B300 over M=1..16 with the same >=5% threshold as the
    # entries above. The replicated projections (2112x7168, 3584x7168,
    # 7168x3584) keep their shapes at TP16 and reuse the entries above, and
    # o_proj lands on 7168x768, which shared_down_proj already covers.
    (3216, 7168): ProjectionSpec(
        3216,
        7168,
        # Both gaps in this range are measured, not oversights: dsv3 is only
        # 4% ahead at M6..M8, and at M16 cuBLAS switches to a faster kernel
        # (11.42us vs dsv3's 11.83us) after trailing it by 6-8% at M9..M15.
        frozenset(range(9, 16)),
        cute_configs=(
            (1, _cute(1, 224, 3, 4)),
            (2, _cute(2, 128, 4, 2)),
            (3, _cute(3, 128, 2, 1)),
            (4, _cute(4, 64, 2, 2)),
            (5, _cute(5, 128, 3, 1)),
        ),
        name="in_proj_qkvgfab",
    ),
    (768, 7168): ProjectionSpec(
        768,
        7168,
        frozenset(range(5, 17)),
        cute_configs=(
            (1, _cute(1, 224, 2, 4)),
            (2, _cute(2, 224, 2, 2)),
            (3, _cute(3, 224, 2, 2)),
            (4, _cute(4, 224, 2, 2)),
        ),
        name="mla_g_proj/shared_gate_up_proj",
    ),
    (1152, 1536): ProjectionSpec(
        1152,
        1536,
        frozenset(range(2, 17)),
        ((1, _cute(1, 192, 3, 4)),),
        name="q_b_proj",
    ),
    (768, 128): ProjectionSpec(768, 128, _M1_TO_16, name="f_b_proj"),
    # dsv3 drops under 5% from M9 on for this shape.
    (7168, 384): ProjectionSpec(
        7168, 384, frozenset(range(1, 9)), name="shared_down_proj"
    ),
    (4224, 7168): ProjectionSpec(
        4224,
        7168,
        frozenset(range(4, 9)),
        cute_configs=(
            (1, _cute(1, 224, 3, 4)),
            (2, _cute(2, 128, 2, 1)),
            (3, _cute(3, 64, 2, 2)),
        ),
        name="dense_gate_up_proj",
    ),
    (10240, 7168): ProjectionSpec(
        10240,
        7168,
        cute_configs=(
            (1, _cute(1, 224, 4, 2)),
            (2, _cute(2, 32, 2, 4)),
            (3, _cute(3, 64, 4, 1)),
            (4, _cute(4, 64, 4, 1)),
        ),
        name="lm_head",
    ),
    # 7168x2112 (TP16 dense down_proj) has no entry on purpose: K=2112 divides
    # none of the fused-A tile_k values, and the CuTe kernel is left with
    # vector_width=2, which measured slower than cuBLAS.
}


def _backend_for(
    spec: ProjectionSpec,
    num_tokens: int,
    has_residual: bool,
    sm121: bool = False,
) -> Backend | None:
    if has_residual:
        if spec.residual_config(num_tokens) is None:
            return None
        if sm121 and ((spec.n, spec.k), num_tokens) not in (
            _SM121_CUTE_RESIDUAL_TOKENS
        ):
            return None
        return "cute"
    if spec.cute_config(num_tokens) is not None and (
        not sm121 or ((spec.n, spec.k), num_tokens) in _SM121_CUTE_TOKENS
    ):
        return "cute"
    if sm121:
        # Authoritative: only measured SM121 winners route, whether or not
        # the SM103 table lists them (K=7168 M9/M16 were unmeasurable there).
        if num_tokens in _SM121_DSV3_TOKENS.get(
            (spec.n, spec.k), frozenset()
        ):
            return "dsv3_fused_a"
        return None
    if num_tokens in spec.dsv3_tokens:
        return "dsv3_fused_a"
    return None


def select_kimi_k3_backend(
    num_tokens: int,
    n: int,
    k: int,
    *,
    has_residual: bool = False,
) -> Backend | None:
    """Backend for a local ``(N, K)`` at ``num_tokens``, or None to fall back."""
    spec = KIMI_K3_PROJECTIONS.get((n, k))
    return _backend_for(spec, num_tokens, has_residual) if spec is not None else None


def _build_plan(
    spec: ProjectionSpec, sm121: bool = False
) -> dict[int, ResolvedCall]:
    plan: dict[int, ResolvedCall] = {}
    for num_tokens in range(1, 17):
        backend = _backend_for(spec, num_tokens, has_residual=False, sm121=sm121)
        if backend == "cute":
            plan[num_tokens] = ("cute", spec.cute_config(num_tokens))
        elif backend == "dsv3_fused_a":
            plan[num_tokens] = ("dsv3_fused_a", None)
    return plan


def _build_residual_plan(
    spec: ProjectionSpec, sm121: bool = False
) -> dict[int, SkinnyGemmConfig]:
    if not sm121:
        return {
            num_tokens: config for num_tokens, config in spec.residual_configs
        }
    return {
        num_tokens: config
        for num_tokens, config in spec.residual_configs
        if ((spec.n, spec.k), num_tokens) in _SM121_CUTE_RESIDUAL_TOKENS
    }


def _is_sm103() -> bool:
    return current_platform.is_device_capability((10, 3))


def _is_sm121() -> bool:
    return current_platform.is_device_capability((12, 1))


# SM121 winners measured 2026-10-04 on GB10 via kernel-validate
# (tools/qualification/validate_dsv3_sm121.py and
# validate_cute_skinny_sm121.py) against cuBLAS, with the same >=1.05x
# threshold the SM103 table uses. Only measured (shape, M) pairs are
# listed; everything else stays on the default GEMM. The DSV3 map is
# authoritative on SM121 (not a filter of the SM103 token sets): K=7168
# shapes were unmeasurable before the smem-budget fix and now win big
# at M9 (shared_gate_up 2.06x, mla_g 1.84x, qkv_a 1.22x).
_SM121_DSV3_TOKENS: dict[tuple[int, int], frozenset[int]] = {
    (2304, 1536): frozenset({1, 9}),  # 1.40x / 1.11x
    (4608, 1536): frozenset({1}),  # 1.21x (M9 loses at 0.82x)
    (7168, 768): frozenset({1, 9}),  # 1.36x / 1.08x
    (1152, 1536): frozenset({1, 9, 16}),  # 1.43x / 1.48x / 1.32x
    (1536, 7168): frozenset({1, 9, 16}),  # 1.43x / 2.06x / 1.77x
    (2112, 7168): frozenset({1, 9}),  # 1.06x / 1.22x (M16 1.045x misses)
    (768, 7168): frozenset({1, 9, 16}),  # 1.98x / 1.84x / 1.58x
    (3216, 7168): frozenset({9}),  # 1.06x (M1/M16 neutral at 1.01x/1.02x)
    (4224, 7168): frozenset({9, 16}),  # 1.07x / 1.06x (M1 neutral at 1.01x)
}
_SM121_CUTE_TOKENS: frozenset[tuple[tuple[int, int], int]] = frozenset({
    ((768, 7168), 1),  # 2.44x (M2-M4 lose at 0.63-0.66x)
    ((1152, 1536), 1),  # 1.90x
    ((3072, 7168), 1),  # 1.11x
    ((3072, 7168), 2),  # 1.10x
    ((3072, 7168), 3),  # 1.06x
    ((3072, 7168), 4),  # 1.10x
    ((3072, 7168), 5),  # 1.07x
    ((3216, 7168), 1),  # 1.06x
    ((3216, 7168), 2),  # 1.06x
    ((3216, 7168), 3),  # 1.06x
    ((3216, 7168), 4),  # 1.13x
    ((3216, 7168), 5),  # 1.07x
    ((3584, 7168), 1),  # 1.06x
    ((4224, 7168), 1),  # 1.06x
    ((6288, 7168), 1),  # 1.06x
    ((7168, 1536), 1),  # 1.85x
    ((7168, 3072), 1),  # 1.46x
    ((7168, 3072), 2),  # 1.07x
    ((7168, 3584), 1),  # 1.47x
    ((7168, 4224), 1),  # 1.05x
    ((8448, 7168), 1),  # 1.06x
    ((12448, 7168), 3),  # 1.06x
})
_SM121_CUTE_RESIDUAL_TOKENS: frozenset[tuple[tuple[int, int], int]] = frozenset({
    ((7168, 3584), 1),  # 1.44x
    ((7168, 3584), 3),  # 1.06x
    ((7168, 3584), 4),  # 1.07x
})


def _is_packed_row_major(tensor: torch.Tensor) -> bool:
    return tensor.dim() == 2 and tensor.stride() == (tensor.shape[1], 1)


def _runtime_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        _is_packed_row_major(x)
        and _is_packed_row_major(weight)
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.shape[1] == weight.shape[1]
    )


def _residual_ok(x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor) -> bool:
    return (
        residual.dim() == 2
        and residual.dtype == torch.bfloat16
        and residual.is_cuda
        and residual.device == x.device
        and residual.is_contiguous()
        and residual.shape == (x.shape[0], weight.shape[0])
    )


def _run_plan(
    plan: dict[int, ResolvedCall], x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor | None:
    entry = plan.get(x.shape[0])
    if entry is None:
        return None
    backend, config = entry
    if backend == "cute":
        if not shape_dynamic_skinny_gemm.is_available():
            return None
        return shape_dynamic_skinny_gemm(x, weight, config, None)
    if not hasattr(torch.ops._C, "dsv3_fused_a_gemm"):
        return None
    output = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    ops.dsv3_fused_a_gemm(output, x, weight.t(), enable_pdl=True)
    return output


def _run_residual_plan(
    residual_plan: dict[int, SkinnyGemmConfig],
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor | None:
    config = residual_plan.get(x.shape[0])
    if config is None or not shape_dynamic_skinny_gemm.is_available():
        return None
    return shape_dynamic_skinny_gemm(x, weight, config, residual)


def try_low_latency_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Run the shape-selected low-latency kernel, or None to fall back.

    Resolves the plan from the shape table on each call; production installs a
    precomputed plan (see :func:`enable_kimi_k3_low_latency_gemm`) and does not
    use this path.
    """
    sm121 = _is_sm121() and not _is_sm103()
    if (
        envs.VLLM_BATCH_INVARIANT
        or not (_is_sm103() or sm121)
        or not _runtime_ok(x, weight)
    ):
        return None
    spec = KIMI_K3_PROJECTIONS.get((weight.shape[0], weight.shape[1]))
    if spec is None:
        return None
    if residual is None:
        return _run_plan(_build_plan(spec, sm121), x, weight)
    if not _residual_ok(x, weight, residual):
        return None
    return _run_residual_plan(
        _build_residual_plan(spec, sm121), x, weight, residual
    )


class _KimiK3LowLatencyApply:
    """Mixin: try the precomputed plan, else defer to the base method."""

    def __init__(self, plan: dict[int, ResolvedCall]) -> None:
        self._plan = plan

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            bias is None
            and not envs.VLLM_BATCH_INVARIANT
            and _runtime_ok(x, layer.weight)
        ):
            output = _run_plan(self._plan, x, layer.weight)
            if output is not None:
                return output
        return super().apply(layer, x, bias)  # type: ignore[misc]


class KimiK3LowLatencyLinearMethod(_KimiK3LowLatencyApply, UnquantizedLinearMethod):
    def __init__(
        self,
        plan: dict[int, ResolvedCall],
        residual_plan: dict[int, SkinnyGemmConfig],
    ) -> None:
        super().__init__(plan)
        self._residual_plan = residual_plan

    def apply_with_residual(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not envs.VLLM_BATCH_INVARIANT
            and _runtime_ok(x, layer.weight)
            and _residual_ok(x, layer.weight, residual)
        ):
            output = _run_residual_plan(self._residual_plan, x, layer.weight, residual)
            if output is not None:
                return output
        return torch.addmm(residual, x, layer.weight.t())


class KimiK3LowLatencyEmbeddingMethod(
    _KimiK3LowLatencyApply, UnquantizedEmbeddingMethod
):
    pass


def enable_kimi_k3_low_latency_gemm(
    module: nn.Module,
    dtype: torch.dtype,
) -> None:
    """Install shape-selected low-latency GEMMs and register CuTe warmups.

    Modules are matched purely by type, an exactly-unquantized method, and a
    local ``(N, K)`` present in :data:`KIMI_K3_PROJECTIONS`.
    """
    sm121 = _is_sm121() and not _is_sm103()
    if dtype != torch.bfloat16 or not (_is_sm103() or sm121):
        return

    warmup_configs: set[SkinnyGemmConfig] = set()
    residual_warmup_configs: set[SkinnyGemmConfig] = set()
    for child in module.modules():
        is_linear = (
            isinstance(child, LinearBase)
            and type(child.quant_method) is UnquantizedLinearMethod
        )
        # ParallelLMHead is a VocabParallelEmbedding subclass; embed_tokens is
        # the parent type, so isinstance already excludes it.
        is_head = (
            isinstance(child, ParallelLMHead)
            and type(child.quant_method) is UnquantizedEmbeddingMethod
        )
        if not (is_linear or is_head):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2:
            continue
        spec = KIMI_K3_PROJECTIONS.get((weight.shape[0], weight.shape[1]))
        if spec is None:
            continue
        if is_linear:
            child.quant_method = KimiK3LowLatencyLinearMethod(
                _build_plan(spec, sm121), _build_residual_plan(spec, sm121)
            )
        else:
            child.quant_method = KimiK3LowLatencyEmbeddingMethod(
                _build_plan(spec, sm121)
            )
        # Warm up only the configs measured for this module's local (N, K) so a
        # TP8 deployment does not compile TP4 configs and vice versa. On SM121
        # warm up only the winner subset so losing configs never JIT.
        if sm121:
            key = (spec.n, spec.k)
            warmup_configs.update(
                config
                for m, config in spec.cute_configs
                if (key, m) in _SM121_CUTE_TOKENS
            )
            residual_warmup_configs.update(
                config
                for m, config in spec.residual_configs
                if (key, m) in _SM121_CUTE_RESIDUAL_TOKENS
            )
        else:
            warmup_configs.update(config for _, config in spec.cute_configs)
            residual_warmup_configs.update(
                config for _, config in spec.residual_configs
            )

    if shape_dynamic_skinny_gemm.is_available():
        if warmup_configs:
            shape_dynamic_skinny_gemm.request_warmup_configs(dtype, warmup_configs)
        if residual_warmup_configs:
            shape_dynamic_skinny_gemm.request_warmup_configs(
                dtype, residual_warmup_configs, has_residual=True
            )
