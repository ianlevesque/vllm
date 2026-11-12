# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X_MLA's binding to standalone and FlashInfer-embedded b12x, without a GPU."""

import sys
import types

import pytest
import torch

from vllm.v1.attention.backends.mla import b12x_mla

MISS = "dense MLA execution is not prepared; resolve launchers before run"


class _Plan:
    def bind(self, **kwargs):
        return ("binding", kwargs)


def _install(monkeypatch, *, embedded: bool, misses: int = 0, error: str = MISS):
    calls: list[str] = []
    state = {"misses": misses}

    def compile_dense_mla(*, binding):
        calls.append("compile")

    def run_dense_mla(*, binding):
        calls.append("run")
        if state["misses"]:
            state["misses"] -= 1
            raise RuntimeError(error)
        return "output", "lse"

    public = types.ModuleType("b12x.attention.dense_mla")
    public.Caps = dict
    if embedded:
        # The embedded public names are session-prepared and must not be used.
        public.plan = public.bind = public.run = lambda *a, **k: pytest.fail("public API")
        kernel = types.ModuleType("b12x.attention.dense_mla._kernel")
        kernel.compile_dense_mla, kernel.run_dense_mla = compile_dense_mla, run_dense_mla
        scratch = types.ModuleType("b12x.attention.dense_mla._scratch")
        scratch.plan_dense_mla_scratch = lambda caps: _Plan()
        public._kernel, public._scratch = kernel, scratch
        monkeypatch.setitem(sys.modules, "b12x.attention.dense_mla._kernel", kernel)
        monkeypatch.setitem(sys.modules, "b12x.attention.dense_mla._scratch", scratch)
    else:
        public.plan = lambda caps: _Plan()
        public.bind = lambda plan, **kwargs: plan.bind(**kwargs)
        public.compile = lambda *, binding: compile_dense_mla(binding=binding)
        public.run = lambda *, binding: run_dense_mla(binding=binding)
    attention = types.ModuleType("b12x.attention")
    attention.dense_mla = public
    root = types.ModuleType("b12x")
    root.attention = attention
    monkeypatch.setitem(sys.modules, "b12x", root)
    monkeypatch.setitem(sys.modules, "b12x.attention", attention)
    monkeypatch.setitem(sys.modules, "b12x.attention.dense_mla", public)
    return public, calls


def test_standalone_b12x_is_used_unchanged(monkeypatch):
    public, _ = _install(monkeypatch, embedded=False)
    assert b12x_mla._load_dense_mla() is public


def test_embedded_b12x_binds_the_scratch_plan_entry_points(monkeypatch):
    public, calls = _install(monkeypatch, embedded=True)
    dense_mla = b12x_mla._load_dense_mla()
    assert dense_mla is not public and dense_mla.Caps is dict
    plan = dense_mla.plan(dense_mla.Caps(mode="decode"))
    binding = dense_mla.bind(plan, q=1, scratch=2)
    assert binding == ("binding", {"q": 1, "scratch": 2})
    dense_mla.compile(binding=binding)
    assert dense_mla.run(binding=binding) == ("output", "lse")
    assert calls == ["compile", "run"]


def test_embedded_launch_miss_compiles_once_outside_capture(monkeypatch):
    _, calls = _install(monkeypatch, embedded=True, misses=1)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    dense_mla = b12x_mla._load_dense_mla()
    assert dense_mla.run(binding="b") == ("output", "lse")
    assert calls == ["run", "compile", "run"]


def test_embedded_launch_miss_stays_fatal_during_capture(monkeypatch):
    _, calls = _install(monkeypatch, embedded=True, misses=1)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="not prepared"):
        b12x_mla._load_dense_mla().run(binding="b")
    assert calls == ["run"]


def test_embedded_second_miss_and_other_errors_propagate(monkeypatch):
    _, calls = _install(monkeypatch, embedded=True, misses=2)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with pytest.raises(RuntimeError, match="not prepared"):
        b12x_mla._load_dense_mla().run(binding="b")
    assert calls == ["run", "compile", "run"]
    _, calls = _install(monkeypatch, embedded=True, misses=1, error="CUDA error")
    with pytest.raises(RuntimeError, match="CUDA error"):
        b12x_mla._load_dense_mla().run(binding="b")
    assert calls == ["run"]
