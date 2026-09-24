"""Save and load activation caches to/from disk, so an expensive forward
pass doesn't need to be re-run for every downstream analysis."""

from __future__ import annotations

from pathlib import Path

import torch


def save_cache(cache: dict[str, torch.Tensor], path: str | Path) -> None:
    """Save an activation cache (as returned by `run_with_cache`) to disk."""
    torch.save(cache, path)


def load_cache(path: str | Path) -> dict[str, torch.Tensor]:
    """Load an activation cache previously saved with `save_cache`."""
    return torch.load(path)
