"""Scoring functions. Each takes (response, example) and returns a float in [0, 1]."""

from __future__ import annotations

import re
from typing import Callable

from eval_harness.tasks import Example

Scorer = Callable[[str, Example], float]

SCORERS: dict[str, Scorer] = {}


def register_scorer(name: str):
    def decorator(fn):
        SCORERS[name] = fn
        return fn

    return decorator


def get_scorer(name: str) -> "Scorer":
    try:
        return SCORERS[name]
    except KeyError:
        raise KeyError(f"Unknown scorer '{name}'. Registered: {sorted(SCORERS)}")


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


@register_scorer("exact_match")
def exact_match(response: str, example: Example) -> float:
    return 1.0 if _normalize(response) == _normalize(example.reference) else 0.0


@register_scorer("multiple_choice")
def multiple_choice(response: str, example: Example) -> float:
    """Score by extracting the first choice letter (A/B/C/...) mentioned in the response."""
    if not example.choices:
        raise ValueError("multiple_choice scorer requires example.choices")

    letters = [chr(ord("A") + i) for i in range(len(example.choices))]
    match = re.search(r"\b([A-Z])\b", response.strip())
    predicted = match.group(1) if match else None
    return 1.0 if predicted == example.reference.strip().upper() and predicted in letters else 0.0


@register_scorer("keyword_rubric")
def keyword_rubric(response: str, example: Example) -> float:
    """Fraction of required keywords (comma-separated in `reference`) present in the response."""
    keywords = [kw.strip() for kw in example.reference.split(",") if kw.strip()]
    if not keywords:
        return 0.0
    normalized_response = _normalize(response)
    hits = sum(1 for kw in keywords if _normalize(kw) in normalized_response)
    return hits / len(keywords)


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _extract_number(text: str) -> float | None:
    """Return the last number mentioned in `text`, or None if there isn't one."""
    matches = _NUMBER.findall(text.replace(",", ""))
    return float(matches[-1]) if matches else None


@register_scorer("numeric_match")
def numeric_match(response: str, example: Example, rel_tol: float = 1e-2) -> float:
    """1.0 if the last number in the response is within `rel_tol` relative
    error of the numeric reference, else 0.0.

    Uses the last number so that worked answers ("3 + 4 = 7") score on the
    final result.
    """
    predicted = _extract_number(response)
    if predicted is None:
        return 0.0
    reference = float(example.reference)
    return 1.0 if abs(predicted - reference) / abs(reference) <= rel_tol else 0.0
