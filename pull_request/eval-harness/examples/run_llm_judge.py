"""Demo: score a couple of responses with an LLM judge (stubbed via
DummyModel so this stays API-free), including the judge's raw reasoning
text to make parsing realistic."""

from __future__ import annotations

from eval_harness.llm_judge import judge_score, judge_score_with_consistency
from eval_harness.model_interface import DummyModel

QUESTION = "Explain why the sky is blue."
RUBRIC = "Award points for mentioning Rayleigh scattering and wavelength dependence."


def main() -> None:
    good_response = (
        "The sky is blue because of Rayleigh scattering: shorter (blue) wavelengths "
        "scatter more strongly in the atmosphere than longer (red) wavelengths."
    )
    mediocre_response = "The sky is blue because of the ocean's reflection."

    cases = {
        "good": (
            good_response,
            "Given the response addresses 2 of the 3 rubric points, Score: 8/10.",
        ),
        "mediocre": (
            mediocre_response,
            "This response misunderstands the mechanism entirely. Score: 12/10.",
        ),
    }

    for label, (response, judge_output) in cases.items():
        # DummyModel returns `default` for any prompt it hasn't seen, which
        # is all we need here since we're stubbing the judge's reply, not
        # testing the prompt text itself.
        judge = DummyModel(default=judge_output)

        score = judge_score(QUESTION, response, RUBRIC, judge)
        print(f"[{label}] judge said {judge_output!r}")
        print(f"[{label}] normalized score: {score:.2f}")

    print()
    consistency_judge = DummyModel(default="Score: 7/10.")
    result = judge_score_with_consistency(QUESTION, good_response, RUBRIC, consistency_judge, n_samples=3)
    print(f"consistency-checked score (n=3): {result!r}")


if __name__ == "__main__":
    main()
