from eval_harness.export import export_results_to_csv
from eval_harness.model_interface import DummyModel
from eval_harness.runner import run_eval
from eval_harness.tasks import Example, Task


def test_export_results_to_csv_writes_one_row_per_example(tmp_path):
    examples = [
        Example(example_id="1", prompt="2+2?", reference="4"),
        Example(example_id="2", prompt="3+3?", reference="6"),
    ]
    task = Task(name="t", scorer_name="exact_match", examples=examples)
    model = DummyModel(responses={"2+2?": "4", "3+3?": "6"})
    result = run_eval(task, model)

    out_path = tmp_path / "results.csv"
    export_results_to_csv(result, out_path)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3  # header + 2 rows
    assert lines[0] == "id,prompt,response,reference,score"
