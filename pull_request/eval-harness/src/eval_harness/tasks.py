"""Task and example data structures, plus a JSONL loader."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Example:
    """A single eval item: a prompt plus everything needed to score a response."""

    example_id: str
    prompt: str
    reference: str
    choices: list[str] | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Task:
    """A named collection of examples plus which scorer to use."""

    name: str
    scorer_name: str
    examples: list[Example]


def load_task_from_jsonl(path: str | Path, name: str, scorer_name: str) -> Task:
    """Load a task from a JSONL file.

    Each line must be a JSON object with keys: id, prompt, reference, and
    optionally choices (list[str]) and metadata (dict).
    """
    examples: list[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(
                Example(
                    example_id=row["id"],
                    prompt=row["prompt"],
                    reference=row["reference"],
                    choices=row.get("choices"),
                    metadata=row.get("metadata", {}),
                )
            )
    return Task(name=name, scorer_name=scorer_name, examples=examples)
