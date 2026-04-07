"""
BIG-Bench Extra Hard benchmark suite.

BIG-Bench Extra Hard (BBH) is a subset of the most challenging tasks from
the BIG-Bench benchmark that language models have historically struggled with.
Tasks include logical reasoning, causal judgment, date understanding, etc.

Reference: https://huggingface.co/datasets/maveriq/bigbenchhard
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .base import BenchmarkResult, BenchmarkSuite

# ---------------------------------------------------------------------------
# Built-in mini dataset (10 representative BIG-Bench Hard problems)
# ---------------------------------------------------------------------------

_BUILTIN_PROBLEMS: list[dict[str, Any]] = [
    {
        "id": "bbh-001",
        "task": "logical_deduction_three_objects",
        "question": "The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph.\n\nOn a shelf, there are three books: a red book, a green book, and a blue book. The blue book is to the left of the green book. The red book is the second from the left.\n\nWhich book is the leftmost?",
        "choices": ["The red book", "The green book", "The blue book"],
        "answer": "C",
    },
    {
        "id": "bbh-002",
        "task": "causal_judgement",
        "question": "How would a typical person answer each of the following questions about causation?\n\nBilly and Suzy each throw a rock at a bottle. Suzy's rock arrives first and shatters the bottle. Billy's rock arrives at where the bottle was a moment later.\n\nDid Billy cause the bottle to shatter?",
        "choices": ["Yes", "No"],
        "answer": "B",
    },
    {
        "id": "bbh-003",
        "task": "date_understanding",
        "question": "Today is Christmas Eve of 1937. What is the date tomorrow in MM/DD/YYYY?",
        "choices": ["12/11/1937", "12/25/1937", "01/04/1938", "12/04/1937", "12/25/2006", "09/25/1937"],
        "answer": "B",
    },
    {
        "id": "bbh-004",
        "task": "disambiguation_qa",
        "question": "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\n\nSentence: The scientist thanked the librarian because she was helpful.\n\nWho was helpful?",
        "choices": ["The scientist", "The librarian", "Ambiguous"],
        "answer": "B",
    },
    {
        "id": "bbh-005",
        "task": "navigate",
        "question": "If you follow these instructions, do you return to the starting point?\n\nTake 1 step east. Take 2 steps north. Take 1 step west. Take 2 steps south.",
        "choices": ["Yes", "No"],
        "answer": "A",
    },
    {
        "id": "bbh-006",
        "task": "penguins_in_a_table",
        "question": "Here is a table where the first line is a header and each subsequent line is a penguin:\n\nname, age, height (cm), weight (kg)\nLouis, 7, 50, 11\nBernard, 5, 80, 13\nVincent, 9, 60, 11\nGwen, 8, 70, 15\n\nWhat is the name of the tallest penguin?",
        "choices": ["Louis", "Bernard", "Vincent", "Gwen"],
        "answer": "B",
    },
    {
        "id": "bbh-007",
        "task": "sports_understanding",
        "question": "Is the following sentence plausible?\n\n\"Lebron James scored a touchdown.\"",
        "choices": ["plausible", "not plausible"],
        "answer": "B",
    },
    {
        "id": "bbh-008",
        "task": "boolean_expressions",
        "question": "Evaluate the result of a random Boolean expression.\n\nnot ( ( not not True ) ) is",
        "choices": ["True", "False"],
        "answer": "B",
    },
    {
        "id": "bbh-009",
        "task": "web_of_lies",
        "question": "Evaluate a random Boolean function expressed as a word problem.\n\nQuestion: Fidel tells the truth. Jerry says Fidel tells the truth. Vina says Jerry lies. Millicent says Vina tells the truth. Does Millicent tell the truth?",
        "choices": ["Yes", "No"],
        "answer": "B",
    },
    {
        "id": "bbh-010",
        "task": "tracking_shuffled_objects_three_objects",
        "question": "Alice, Bob, and Claire are playing a game. At the start of the game, they are each holding a ball: Alice has a red ball, Bob has a blue ball, and Claire has a green ball.\n\nAs the game progresses, pairs of players trade balls. First, Alice and Bob swap balls. Then, Bob and Claire swap balls.\n\nAt the end of the game, what ball does Bob have?",
        "choices": ["red ball", "blue ball", "green ball"],
        "answer": "C",
    },
]


class BigBenchHardBenchmark(BenchmarkSuite):
    """BIG-Bench Extra Hard: Challenging reasoning tasks from BIG-Bench."""

    name = "BigBench-Extra-Hard"
    description = "Challenging multi-step reasoning tasks from BIG-Bench"
    category = "reasoning"

    def load_dataset(self) -> list[dict[str, Any]]:
        jsonl_path = Path(self.data_dir) / "bigbench_hard.jsonl"
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
        choices = problem.get("choices", [])
        letters = "ABCDEFGHIJ"
        choices_text = "\n".join(
            f"  {letters[i]}. {c}" for i, c in enumerate(choices)
        )
        return (
            f"Answer the following reasoning question.\n\n"
            f"Question: {question}\n\n"
            f"Choices:\n{choices_text}\n\n"
            f"Think step by step, then write ONLY the letter of the correct answer "
            f"to a file called answer.txt — no explanation, just the single letter."
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
        match = re.search(r"\b([A-J])\b", agent_output)
        if match:
            answer_path.write_text(match.group(1) + "\n", encoding="utf-8")
            metadata["recovered_answer_from_output"] = True

    def evaluate(self, problem: dict[str, Any], workspace: str) -> BenchmarkResult:
        pid = problem.get("id", "unknown")
        expected = str(problem["answer"]).strip().upper()
        answer_file = os.path.join(workspace, "answer.txt")

        if not os.path.exists(answer_file):
            return BenchmarkResult(
                problem_id=pid, passed=False, expected=expected,
                error="answer.txt not found",
            )

        with open(answer_file) as fh:
            actual_raw = fh.read().strip()

        match = re.search(r"\b([A-J])\b", actual_raw.upper())
        actual = match.group(1) if match else actual_raw.strip().upper()

        passed = actual == expected
        return BenchmarkResult(
            problem_id=pid, passed=passed,
            expected=expected, actual=actual_raw,
            error="" if passed else f"expected={expected}, got={actual_raw}",
        )
