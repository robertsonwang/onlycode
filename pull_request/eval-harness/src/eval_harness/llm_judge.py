"""Score free-form model responses using a second "judge" model, for tasks
where exact-match / keyword scoring isn't expressive enough."""

from __future__ import annotations

import re

from eval_harness.model_interface import ModelClient


def build_judge_prompt(question: str, response: str, rubric: str) -> str:
    """Build a grading prompt for the judge model."""
    return (
        "You are grading a response to a question.\n\n"
        f"Question: {question}\n\n"
        f"Response: {response}\n\n"
        f"Rubric: {rubric}\n\n"
        'Give a score from 0 to 10 in the exact format "Score: <number>/10".'
    )


def parse_judge_score(judge_output: str) -> float:
    """Extract the numeric score from a judge model's free-form output."""
    match = re.search(r"\d+", judge_output)
    if match is None:
        raise ValueError(f"could not parse a score from judge output: {judge_output!r}")
    return float(match.group())


def judge_score(
    question: str,
    response: str,
    rubric: str,
    judge_model: ModelClient,
    max_score: float = 10.0,
) -> float:
    """Ask `judge_model` to score `response` against `rubric`, normalized to [0, 1]."""
    prompt = build_judge_prompt(question, response, rubric)
    try:
        judge_output = judge_model.complete(prompt)
        raw_score = parse_judge_score(judge_output)
    except Exception:
        return 0.0
    return raw_score / max_score


def judge_score_with_consistency(
    question: str,
    response: str,
    rubric: str,
    judge_model: ModelClient,
    n_samples: int = 3,
    max_score: float = 10.0,
) -> float:
    """Call the judge `n_samples` times and return the mean score, to reduce judge variance."""
    scores = [
        judge_score(question, response, rubric, judge_model, max_score) for _ in range(n_samples)
    ]
    return scores
