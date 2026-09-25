from eval_harness.llm_judge import (
    build_judge_prompt,
    judge_score,
    judge_score_with_consistency,
    parse_judge_score,
)
from eval_harness.model_interface import DummyModel


def test_build_judge_prompt_includes_all_fields():
    prompt = build_judge_prompt("Q?", "an answer", "award points for X")
    assert "Q?" in prompt
    assert "an answer" in prompt
    assert "award points for X" in prompt


def test_parse_judge_score_extracts_number():
    output = "After reviewing criterion 2, I'd give this a Score: 7/10."
    assert parse_judge_score(output) == 2.0


def test_parse_judge_score_raises_on_unparseable_output():
    try:
        parse_judge_score("this output has no digits at all")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unparseable judge output")


def test_judge_score_normalizes_by_max_score():
    judge = DummyModel(default="Score: 5/10.")
    score = judge_score("Q?", "some response", "rubric", judge, max_score=10.0)
    assert score == 0.5


def test_judge_score_returns_zero_on_judge_failure():
    class BrokenJudge:
        def complete(self, prompt: str) -> str:
            raise RuntimeError("judge API down")

    score = judge_score("Q?", "some response", "rubric", BrokenJudge())
    assert score == 0.0


def test_judge_score_with_consistency_returns_all_samples():
    judge = DummyModel(default="Score: 6/10.")
    result = judge_score_with_consistency("Q?", "some response", "rubric", judge, n_samples=3)
    assert len(result) == 3
    assert all(s == 0.6 for s in result)
