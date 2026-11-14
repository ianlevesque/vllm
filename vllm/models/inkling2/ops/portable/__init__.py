# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable Triton kernels for Inkling2 on GPUs without the SM10x paths
(tml_fa4 paged KV, Gluon tcgen05 decode, TRT-LLM-gen expert GEMMs): the
relative-bias attention and block-scaled NVFP4/MXFP8 experts of TokenSpeed
(MIT, vendored), used on SM12x such as DGX Spark (GB10)."""
