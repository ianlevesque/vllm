#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Finite installed-image MiMo sink gate for wim kernel-validate.

Twelve dtype/shape cases, exactly three attention invocations per case on success.
Additional Torch reference, copy, guard and reduction kernels are not counted.
No model requests, downloads, source overrides, pytest skip or distributed claim.
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["mimo-vision-sink"], required=True)
    parser.add_argument("--expected-vllm-commit", required=True)
    args = parser.parse_args()
    source = Path("/opt/kimi-k3/vllm")
    sha = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    assert sha == args.expected_vllm_commit, (sha, args.expected_vllm_commit)
    relatives = [
        "tests/standalone/run_mimo_vision_sink_native.py",
        "tests/standalone/mimo_vision_sink_reference.py",
        "tests/models/multimodal/test_mimo_v2_omni.py",
        "vllm/model_executor/models/mimo_v2_omni.py",
        "vllm/v1/attention/ops/triton_prefill_attention.py",
        "tests/standalone/mimo_vision_sink_guards.py",
    ]
    hashes, source_bytes = {}, {}
    for relative in relatives:
        raw = (source / relative).read_bytes()
        committed = subprocess.check_output(
            ["git", "-C", str(source), "show", f"HEAD:{relative}"]
        )
        assert raw == committed, (relative, "installed source differs from commit")
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        source_bytes[relative] = raw
    assert Path(__file__).read_bytes() == (source / relatives[0]).read_bytes()
    sys.path.insert(0, str(source))
    import pytest
    import torch

    import vllm

    assert Path(vllm.__file__).resolve().is_relative_to(source)
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 1)
    assert torch.cuda.is_bf16_supported()
    spec = importlib.util.spec_from_file_location(
        "mimo_sink_native", source / relatives[2]
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert len(module.CASES) == 6
    print(
        json.dumps(
            {
                "source": sha,
                "hashes": hashes,
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(),
                "suite": args.suite,
            }
        ),
        flush=True,
    )
    count = 0
    try:
        for dtype in (torch.bfloat16, torch.float16):
            for case in module.CASES:
                with pytest.MonkeyPatch.context() as monkeypatch:
                    result = module.run_window_case(monkeypatch, dtype, case)
                count += 1
                print(json.dumps({"case_passed": count, **result}), flush=True)
    finally:
        torch.cuda.synchronize()
        for relative, raw in source_bytes.items():
            assert (source / relative).read_bytes() == raw, (
                relative,
                "source changed during native run",
            )
        assert Path(__file__).read_bytes() == source_bytes[relatives[0]]
        print(
            json.dumps({"source_bytes_unchanged": True, "device_synchronized": True}),
            flush=True,
        )
    assert count == 12
    print(
        json.dumps(
            {
                "passed": 12,
                "failed": 0,
                "skipped": 0,
                "attention_invocations": 36,
                "additional_torch_reference_and_guard_kernels": True,
                "real_tp_communication": False,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
