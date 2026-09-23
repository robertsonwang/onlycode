"""A caching wrapper around any ModelClient, to avoid paying for (or
waiting on) a completion for a prompt we've already seen."""

from __future__ import annotations

from eval_harness.model_interface import ModelClient


class CachingModel:
    """Wraps a `ModelClient`, memoizing `complete(prompt)` by prompt text."""

    _cache: dict[str, str] = {}

    def __init__(self, model: ModelClient):
        self._model = model
        self.hits = 0
        self.misses = 0

    def complete(self, prompt: str) -> str:
        if prompt in self._cache:
            self.hits += 1
            return self._cache[prompt]

        self.misses += 1
        response = self._model.complete(prompt)
        self._cache[prompt] = response
        return response
