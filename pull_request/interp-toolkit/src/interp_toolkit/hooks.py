"""Hook-point machinery for caching and patching activations.

A `HookPoint` is a transparent nn.Module: on a plain forward pass it is the
identity function. Callers attach temporary functions to it (via
`run_with_cache` / `run_with_patch`) to observe or overwrite the tensor that
flows through that point in the computation graph.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

import torch
import torch.nn as nn

HookFn = Callable[[torch.Tensor, str], torch.Tensor]


class HookPoint(nn.Module):
    """A named pass-through point in the computation graph."""

    def __init__(self) -> None:
        super().__init__()
        self.name: str | None = None
        self._hook: HookFn | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._hook is not None:
            return self._hook(x, self.name or "")
        return x


def _named_hook_points(model: nn.Module) -> dict[str, HookPoint]:
    if hasattr(model, "named_hook_points"):
        return model.named_hook_points()
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, HookPoint)
    }


@contextmanager
def _install_hooks(
    model: nn.Module, hooks: dict[str, HookFn]
) -> Iterator[None]:
    points = _named_hook_points(model)
    unknown = set(hooks) - set(points)
    if unknown:
        raise KeyError(f"Unknown hook point(s): {sorted(unknown)}")

    installed = []
    for name, point in points.items():
        point.name = name
        if name in hooks:
            point._hook = hooks[name]
            installed.append(point)
    try:
        yield
    finally:
        for point in installed:
            point._hook = None


def run_with_cache(
    model: nn.Module, tokens: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run a forward pass, returning the logits and every hook point's activation."""
    cache: dict[str, torch.Tensor] = {}

    def make_cacher(name: str) -> HookFn:
        def cacher(x: torch.Tensor, hook_name: str) -> torch.Tensor:
            cache[name] = x.detach().clone()
            return x

        return cacher

    points = _named_hook_points(model)
    hooks = {name: make_cacher(name) for name in points}
    with _install_hooks(model, hooks):
        logits = model(tokens)
    return logits, cache


def run_with_patch(
    model: nn.Module,
    tokens: torch.Tensor,
    patches: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run a forward pass, overwriting the named hook points with `patches`."""

    def make_patcher(value: torch.Tensor) -> HookFn:
        def patcher(x: torch.Tensor, hook_name: str) -> torch.Tensor:
            return value

        return patcher

    hooks = {name: make_patcher(value) for name, value in patches.items()}
    with _install_hooks(model, hooks):
        logits = model(tokens)
    return logits
