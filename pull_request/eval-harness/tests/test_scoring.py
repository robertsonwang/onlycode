from eval_harness.scoring import exact_match, keyword_rubric, multiple_choice, numeric_tolerance
from eval_harness.tasks import Example


def test_exact_match_is_case_and_whitespace_insensitive():
    ex = Example(example_id="1", prompt="p", reference="  Paris ")
    assert exact_match("paris", ex) == 1.0
    assert exact_match("PARIS", ex) == 1.0
    assert exact_match("London", ex) == 0.0


def test_multiple_choice_extracts_letter():
    ex = Example(
        example_id="1",
        prompt="p",
        reference="B",
        choices=["Oxygen", "Carbon Dioxide", "Nitrogen", "Hydrogen"],
    )
    assert multiple_choice("The answer is B", ex) == 1.0
    assert multiple_choice("I think it's C", ex) == 0.0


def test_multiple_choice_requires_choices():
    ex = Example(example_id="1", prompt="p", reference="B")
    try:
        multiple_choice("B", ex)
    except ValueError:
        return
    raise AssertionError("expected ValueError when choices is None")


def test_keyword_rubric_partial_credit():
    ex = Example(example_id="1", prompt="p", reference="gradient, backprop, chain rule")
    response = "We use the chain rule and compute gradients during backprop."
    score = keyword_rubric(response, ex)
    assert score == 1.0

    partial = keyword_rubric("We use the chain rule.", ex)
    assert abs(partial - 1 / 3) < 1e-9


def test_numeric_tolerance_accepts_close_values():
    ex = Example(example_id="1", prompt="p", reference="60")
    assert numeric_tolerance("60", ex) == 1.0
    assert numeric_tolerance("59.9", ex) == 0.0


def test_numeric_tolerance_respects_metadata_override():
    ex = Example(example_id="1", prompt="p", reference="60", metadata={"tolerance": 1.0})
    assert numeric_tolerance("60.5", ex) == 1.0
