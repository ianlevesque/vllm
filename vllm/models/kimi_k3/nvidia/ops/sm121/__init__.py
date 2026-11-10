# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 fused kernels for SM12x (GB10 / DGX Spark).

SM12x counterparts of the SM100-only fused kernels on the Kimi-K3 decode path:
fused KDA decode and speculative decode, AttnRes, and the cross-node (RoCE)
forms of GEMM + all-reduce and the latent-MoE tail fusion.
"""
