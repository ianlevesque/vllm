# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X >= 1.5 dense-MLA plans are preparation declarations: geometry
(caps, splits, scratch specs) lives on the prepared state's layout, and
the backend must resolve it instead of reading the declaration."""

from types import SimpleNamespace

from vllm.v1.attention.backends.mla import b12x_mla


class _DeclarationPlan:
    """Stands in for b12x.preparation.Plan: no geometry attributes."""


def _layout() -> SimpleNamespace:
    return SimpleNamespace(
        num_splits=8,
        chunks_per_split=4,
        caps=SimpleNamespace(max_page_table_width=77),
        shapes_and_dtypes=lambda: (((128,), "uint8"),),
    )


def test_dense_mla_layout_resolves_and_caches_state(monkeypatch) -> None:
    layout = _layout()
    calls: list[object] = []

    def fake_state(plan, device=None):
        calls.append(plan)
        return SimpleNamespace(layout=layout)

    monkeypatch.setattr(
        b12x_mla, "_load_dense_mla", lambda: SimpleNamespace(state=fake_state)
    )
    declaration = _DeclarationPlan()
    assert b12x_mla._dense_mla_layout(declaration) is layout
    assert b12x_mla._dense_mla_layout(declaration) is layout
    assert calls == [declaration]

    assert b12x_mla._active_dense_mla_splits(declaration, None) == 8
    assert b12x_mla._active_dense_mla_splits(declaration, 64) == 1
    assert b12x_mla._active_dense_mla_splits(declaration, 64 * 9) == 3


def test_dense_mla_layout_passes_native_plans_through(monkeypatch) -> None:
    def forbidden_state(plan, device=None):  # pragma: no cover - failure path
        raise AssertionError("native plans must not be re-resolved")

    monkeypatch.setattr(
        b12x_mla, "_load_dense_mla", lambda: SimpleNamespace(state=forbidden_state)
    )
    layout = _layout()
    assert b12x_mla._dense_mla_layout(layout) is layout
