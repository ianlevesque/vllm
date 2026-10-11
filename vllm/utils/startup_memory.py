# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in startup bookends without device queries or synchronization.

Allocator counters are CPU-side and their peaks are cumulative. These markers
delimit Python calls, not completion of asynchronously queued GPU operations.
"""

import json
import os
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def _emit(stage: str, event: str, **fields: Any) -> None:
    # Observation must not replace an engine exception or fail startup.
    with suppress(Exception):
        logger.info(
            "Startup memory diagnostic %s",
            json.dumps({"stage": stage, "event": event, "pid": os.getpid(), **fields}),
        )


def _proc_memory() -> dict[str, int]:
    result = {}
    for path, keys in (
        ("/proc/self/status", {"VmRSS", "RssAnon", "RssShmem", "VmPin"}),
        ("/proc/meminfo", {"MemAvailable", "Shmem"}),
    ):
        for line in Path(path).read_text().splitlines():
            name, _, value = line.partition(":")
            if name in keys:
                result[name + "_bytes"] = int(value.split()[0]) * 1024
    return result


def _snapshot(stage: str, when: str, device: torch.device) -> None:
    # Isolate readers: an unavailable /proc field should not hide allocator data.
    for name, read in (
        ("proc", _proc_memory),
        ("allocator", lambda: torch.cuda.memory_stats(device)),
    ):
        try:
            values = read()
            if name == "allocator":
                values = {
                    key: values.get(key)
                    for key in (
                        "allocated_bytes.all.current",
                        "allocated_bytes.all.peak",
                        "reserved_bytes.all.current",
                        "reserved_bytes.all.peak",
                    )
                }
            _emit(stage, "snapshot", when=when, reader=name, values=values)
        except Exception as error:
            _emit(
                stage,
                "snapshot_error",
                when=when,
                reader=name,
                error=type(error).__name__,
            )


def _tensor_metadata(inputs: Any) -> dict[str, Any]:
    """Bounded shape-only walk of MM kwargs/field data; never retain tensors."""
    tensors: list[dict[str, Any]] = []
    remaining = 128

    def visit(value: Any, path: str, depth: int = 0) -> None:
        nonlocal remaining
        if remaining <= 0 or depth > 8:
            return
        remaining -= 1
        if isinstance(value, torch.Tensor):
            tensors.append(
                {
                    "path": path,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "bytes": value.numel() * value.element_size(),
                }
            )
        elif isinstance(value, Mapping):
            for key, child in value.items():
                if remaining <= 0:
                    break
                visit(child, f"{path}.{key}", depth + 1)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                if remaining <= 0:
                    break
                visit(child, f"{path}[{index}]", depth + 1)
        elif type(value).__name__ == "MultiModalFieldElem":
            visit(value.data, path + ".data", depth + 1)

    visit(inputs, "inputs")
    return {"tensors": tensors, "truncated": remaining <= 0}


def startup_profile_metadata(runner: Any, inputs: Any = None) -> dict[str, Any]:
    """Read already-resolved scalar geometry and optional tensor metadata."""
    model = getattr(runner, "model", None)
    vision = None
    vision_path = None
    # Native Gemma owns its tower directly. The Transformers wrapper owns an
    # AutoModel at .model, optionally with a conditional-generation wrapper.
    for path in ("vision_tower", "model.vision_tower", "model.model.vision_tower"):
        candidate = model
        for name in path.split("."):
            candidate = getattr(candidate, name, None)
        if candidate is not None:
            vision, vision_path = candidate, path
            break
    config = getattr(runner, "vllm_config", None)
    model_config = getattr(config, "model_config", None)
    compilation = getattr(config, "compilation_config", None)
    backend = getattr(getattr(config, "attention_config", None), "backend", None)
    graph_mode = getattr(compilation, "cudagraph_mode", None)
    speculator = getattr(runner, "speculator", None)
    layers = getattr(compilation, "static_forward_context", {}).values()
    backend_classes = {
        type(layer.impl).__name__ for layer in layers if hasattr(layer, "impl")
    }
    metadata = {
        "model": type(model).__name__,
        "max_num_tokens": getattr(runner, "max_num_tokens", None),
        "max_num_reqs": getattr(runner, "max_num_reqs", None),
        "max_model_len": getattr(model_config, "max_model_len", None),
        "attention_backend": str(backend) if backend is not None else None,
        "attention_impls": sorted(backend_classes),
        "cudagraph_mode": str(graph_mode) if graph_mode is not None else None,
        "vision_model": type(vision).__name__,
        "vision_path": vision_path,
        "vision_attention_backend": getattr(
            getattr(vision, "config", None), "_attn_implementation", None
        ),
        "draft": {
            "class": type(speculator).__name__,
            **{
                name: getattr(speculator, name, None)
                for name in (
                    "num_speculative_steps",
                    "num_query_per_req",
                    "num_extra_query_per_req",
                    "hidden_size",
                    "draft_max_seq_len",
                )
            },
        },
    }
    if inputs is not None:
        metadata["inputs"] = _tensor_metadata(inputs)
        if isinstance(inputs, (list, tuple)):
            metadata["input_items"] = len(inputs)
            metadata["input_modalities"] = sorted(
                {
                    item[0]
                    for item in inputs[:128]
                    if isinstance(item, tuple) and item and isinstance(item[0], str)
                }
            )
    return metadata


@contextmanager
def startup_memory_stage(
    stage: str,
    device: torch.device,
    *,
    enabled: bool = True,
    metadata: Callable[[], dict[str, Any]] | None = None,
) -> Iterator[None]:
    if not enabled or not envs.VLLM_STARTUP_MEMORY_DIAGNOSTICS:
        metadata = None
        yield
        return

    start = time.monotonic()
    # Persist entry BEFORE optional metadata/allocator reads, which may stall.
    _emit(stage, "begin", observation="python_call_boundary_no_cuda_sync")
    if metadata is not None:
        try:
            _emit(stage, "metadata", values=metadata())
        except Exception as error:
            _emit(stage, "metadata_error", error=type(error).__name__)
    # Drop closures before the body so diagnostics do not extend tensor lifetime.
    metadata = None
    _snapshot(stage, "before", device)
    try:
        yield
    except BaseException as error:
        _emit(
            stage,
            "end",
            status="error",
            error=type(error).__name__,
            elapsed_s=time.monotonic() - start,
        )
        # Do not acquire allocator locks while propagating an engine failure.
        raise
    else:
        _emit(stage, "end", status="ok", elapsed_s=time.monotonic() - start)
        _snapshot(stage, "after", device)
