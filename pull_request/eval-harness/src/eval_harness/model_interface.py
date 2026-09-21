"""Model client abstraction. Swap `DummyModel` for a real API-backed client."""

from __future__ import annotations

from typing import Protocol


class ModelClient(Protocol):
    def complete(self, prompt: str) -> str:
        """Return a single text completion for `prompt`."""
        ...


class DummyModel:
    """A deterministic stand-in model, useful for testing the harness itself.

    `responses` maps prompt -> canned completion. Any prompt not in the map
    returns `default`.
    """

    def __init__(self, responses: dict[str, str] | None = None, default: str = ""):
        self.responses = responses or {}
        self.default = default
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.responses.get(prompt, self.default)
