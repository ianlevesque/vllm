# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JIT build of the SM12x fused-kernel extension (same mechanism as topk16)."""

import hashlib
import os
from functools import lru_cache
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SOURCE = "sm121_fused_ext.cu"
_HEADERS = (
    "sm121_common.cuh",
    "kda_decode.cuh",
    "attn_res.cuh",
    "probe.cuh",
    "roce_fused.cuh",
    "roce_gemm.cuh",
)


def source_digest() -> str:
    """Digest of the translation unit and every header it includes.

    torch's JIT versioner hashes only the listed sources, so the digest goes
    into the module name: a header edit must never reuse a stale build.
    """
    digest = hashlib.sha256()
    for name in (_SOURCE, *_HEADERS):
        digest.update(name.encode())
        digest.update((_HERE / name).read_bytes())
    return digest.hexdigest()[:12]


@lru_cache(maxsize=1)
def load_extension():
    from torch.utils.cpp_extension import load

    build_root = os.environ.get("VLLM_KIMI_SM121_EXT_DIR")
    name = f"vllm_kimi_sm121_fused_{source_digest()}"
    build_directory = None
    if build_root:
        build_directory = os.path.join(build_root, name)
        os.makedirs(build_directory, exist_ok=True)
    # VLLM_KIMI_SM121_EXT_VERBOSE=1 prints the build log, including ptxas'
    # per-kernel register and stack usage (a non-zero stack frame means the
    # compiler spilled a kernel's working set to local memory).
    verbose = os.environ.get("VLLM_KIMI_SM121_EXT_VERBOSE") == "1"
    cuda_flags = ["-O3"] + (["-Xptxas", "-v"] if verbose else [])
    return load(
        name=name,
        sources=[str(_HERE / _SOURCE)],
        extra_include_paths=[str(_HERE)],
        extra_cuda_cflags=cuda_flags,
        build_directory=build_directory,
        verbose=verbose,
    )
