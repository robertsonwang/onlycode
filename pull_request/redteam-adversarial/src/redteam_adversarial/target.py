"""A toy bag-of-words safety classifier, used as an attack target.

This stands in for a real content-moderation / safety model: it scores text
by how many words from a fixed "unsafe" vocabulary it contains, normalized
to a probability via a logistic function. It's deterministic and dependency-free
so attack-search experiments run instantly and reproducibly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Protocol

_DEFAULT_UNSAFE_WORDS = {
    "bomb": 3.0,
    "weapon": 2.0,
    "hack": 1.5,
    "exploit": 1.5,
    "poison": 2.5,
    "attack": 1.0,
    "kill": 2.0,
    "steal": 1.0,
}


class SafetyClassifier(Protocol):
    """Anything that maps text to P(unsafe) can be attacked."""

    def unsafe_probability(self, text: str) -> float: ...


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


@dataclass
class ToySafetyClassifier:
    """score(text) -> P(unsafe), higher means the classifier flags it as unsafe."""

    word_weights: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_UNSAFE_WORDS))
    bias: float = -1.0

    def raw_score(self, text: str) -> float:
        tokens = tokenize(text)
        return self.bias + sum(self.word_weights.get(tok, 0.0) for tok in tokens)

    def unsafe_probability(self, text: str) -> float:
        z = self.raw_score(text)
        return 1.0 / (1.0 + math.exp(-z))

    def is_unsafe(self, text: str, threshold: float = 0.5) -> bool:
        return self.unsafe_probability(text) >= threshold
