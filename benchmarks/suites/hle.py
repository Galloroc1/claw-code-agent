"""
HLE (Humanity's Last Exam) benchmark suite.

HLE is an extremely challenging benchmark designed to test the limits of
AI capabilities. It contains questions from diverse domains that are
intended to be among the hardest questions answerable by humans.

Reference: https://huggingface.co/datasets/cais/hle
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .base import BenchmarkResult, BenchmarkSuite

# ---------------------------------------------------------------------------
# Built-in mini dataset (10 representative HLE-style problems)
# ---------------------------------------------------------------------------

_BUILTIN_PROBLEMS: list[dict[str, Any]] = [
    {
        "id": "hle-001",
        "subject": "mathematics",
        "question": "What is the smallest positive integer n such that n! ends with exactly 100 trailing zeros?",
        "answer_type": "exact",
        "answer": "405",
    },
    {
        "id": "hle-002",
        "subject": "physics",
        "question": "In natural units (ℏ = c = 1), the fine-structure constant α ≈ 1/137. What is the approximate ratio of the electromagnetic force to the gravitational force between two protons?",
        "answer_type": "multiple_choice",
        "choices": ["10^36", "10^24", "10^42", "10^18"],
        "answer": "A",
    },
    {
        "id": "hle-003",
        "subject": "computer_science",
        "question": "What is the time complexity of the best known algorithm for matrix multiplication as of 2024?",
        "answer_type": "multiple_choice",
        "choices": ["O(n^2.371552)", "O(n^2.5)", "O(n^3)", "O(n^2 log n)"],
        "answer": "A",
    },
    {
        "id": "hle-004",
        "subject": "mathematics",
        "question": "How many groups of order 16 are there up to isomorphism?",
        "answer_type": "exact",
        "answer": "14",
    },
    {
        "id": "hle-005",
        "subject": "chemistry",
        "question": "What is the maximum number of stereoisomers possible for a molecule with 3 chiral centers and no meso forms?",
        "answer_type": "exact",
        "answer": "8",
    },
    {
        "id": "hle-006",
        "subject": "biology",
        "question": "The human genome contains approximately how many protein-coding genes?",
        "answer_type": "multiple_choice",
        "choices": ["~5,000", "~20,000", "~100,000", "~500,000"],
        "answer": "B",
    },
    {
        "id": "hle-007",
        "subject": "mathematics",
        "question": "What is the value of the Ramanujan sum c_5(3)?",
        "answer_type": "exact",
        "answer": "1",
    },
    {
        "id": "hle-008",
        "subject": "physics",
        "question": "What is the spin of the Higgs boson?",
        "answer_type": "exact",
        "answer": "0",
    },
    {
        "id": "hle-009",
        "subject": "computer_science",
        "question": "In computational complexity theory, which of the following containment relationships is known to be strict?",
        "answer_type": "multiple_choice",
        "choices": ["P ⊂ NP", "AC0 ⊂ NC1", "NP ⊂ PSPACE", "L ⊂ NL"],
        "answer": "B",
    },
    {
        "id": "hle-010",
        "subject": "mathematics",
        "question": "What is the chromatic number of the Petersen graph?",
        "answer_type": "exact",
        "answer": "3",
    },
]


class HLEBenchmark(BenchmarkSuite):
    """HLE: Humanity's Last Exam — extremely challenging expert questions."""

    name = "HLE"
    description = "Extremely challenging expert-level questions (Humanity's Last Exam)"
    category = "knowledge"

    def load_dataset(self) -> list[dict[str, Any]]:
        jsonl_path = Path(self.data_dir) / "hle.jsonl"
        if jsonl_path.exists():
            problems: list[dict[str, Any]] = []
            with open(jsonl_path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        problems.append(json.loads(line))
            if self.verbose:
                print(f"  Loaded {len(problems)} problems from {jsonl_path}")
            return problems

        if self.verbose:
            print(f"  {jsonl_path} not found — using built-in 10-problem subset")
        return list(_BUILTIN_PROBLEMS)

    def build_prompt(self, problem: dict[str, Any]) -> str:
        question = problem["question"]
        answer_type = problem.get("answer_type", "exact")

        if answer_type == "multiple_choice" and "choices" in problem:
            choices = problem["choices"]
            letters = "ABCD"
            choices_text = "\n".join(
                f"  {letters[i]}. {c}" for i, c in enumerate(choices)
            )
            return (
                f"Answer the following extremely challenging question.\n\n"
                f"Question: {question}\n\n"
                f"Choices:\n{choices_text}\n\n"
                f"Think very carefully. Write ONLY the letter of the correct answer "
                f"to a file called answer.txt — no explanation, just the single letter."
            )
        else:
            return (
                f"Answer the following extremely challenging question.\n\n"
                f"Question: {question}\n\n"
                f"Think very carefully. Write ONLY the final answer "
                f"to a file called answer.txt — no explanation, just the answer."
            )

    def recover_output_files(
        self,
        problem: dict[str, Any],
        workspace: str,
        agent_output: str,
        metadata: dict[str, Any],
    ) -> None:
        del problem
        answer_path = Path(workspace) / "answer.txt"
        if answer_path.exists():
            return
        lines = agent_output.strip().splitlines()
        if lines:
            answer_path.write_text(lines[-1].strip() + "\n", encoding="utf-8")
            metadata["recovered_answer_from_output"] = True

    def evaluate(self, problem: dict[str, Any], workspace: str) -> BenchmarkResult:
        pid = problem.get("id", "unknown")
        expected = str(problem["answer"]).strip()
        answer_type = problem.get("answer_type", "exact")
        answer_file = os.path.join(workspace, "answer.txt")

        if not os.path.exists(answer_file):
            return BenchmarkResult(
                problem_id=pid, passed=False, expected=expected,
                error="answer.txt not found",
            )

        with open(answer_file) as fh:
            actual_raw = fh.read().strip()

        if answer_type == "multiple_choice":
            match = re.search(r"\b([A-D])\b", actual_raw.upper())
            actual = match.group(1) if match else actual_raw.strip().upper()
            passed = actual == expected.upper()
        else:
            # Exact match — normalize numbers
            actual = actual_raw.strip()
            try:
                passed = float(actual) == float(expected)
            except (ValueError, TypeError):
                passed = actual.lower() == expected.lower()

        return BenchmarkResult(
            problem_id=pid, passed=passed,
            expected=expected, actual=actual_raw,
            error="" if passed else f"expected={expected}, got={actual_raw}",
        )
