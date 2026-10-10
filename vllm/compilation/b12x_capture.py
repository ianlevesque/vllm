# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_b12x_compile_only_warmup: ContextVar[bool] = ContextVar(
    "b12x_compile_only_warmup",
    default=False,
)


def is_b12x_compile_only_warmup() -> bool:
    """Return whether B12X kernels may compile but must not launch."""
    return _b12x_compile_only_warmup.get()


@contextmanager
def b12x_compile_only_warmup() -> Iterator[None]:
    """Resolve B12X bindings without launching their compute kernels."""
    token = _b12x_compile_only_warmup.set(True)
    try:
        yield
    finally:
        _b12x_compile_only_warmup.reset(token)
