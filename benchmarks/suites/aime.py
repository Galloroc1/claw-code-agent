"""
AIME benchmark suite.

AIME (American Invitational Mathematics Examination) problems are
challenging high-school competition problems.  All answers are integers
from 000 to 999.

Reference: https://artofproblemsolving.com/wiki/index.php/AIME
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .base import BenchmarkResult, BenchmarkSuite

# ---------------------------------------------------------------------------
# Built-in mini dataset (10 representative AIME-style problems)
# ---------------------------------------------------------------------------

_BUILTIN_PROBLEMS: list[dict[str, Any]] = [
    {
        "id": "aime-001",
        "problem": "Find the sum of all positive integers $n$ such that $n^2 - 19n + 99$ is a perfect square.",
        "answer": "38",
    },
    {
        "id": "aime-002",
        "problem": "The number $2^{1993} + 3^{1993}$ is divisible by 5. What is the units digit of the quotient $\\frac{2^{1993}+3^{1993}}{5}$?",
        "answer": "3",
    },
    {
        "id": "aime-003",
        "problem": "Let $S$ be the set of integers between 1 and $2^{40}$ whose binary expansions have exactly two 1's. If a number is chosen at random from $S$, what is the probability that it is divisible by 9? Express as $p/q$ where $p$ and $q$ are coprime, and find $p+q$.",
        "answer": "913",
    },
    {
        "id": "aime-004",
        "problem": "What is the largest prime factor of $1{,}000{,}027$?",
        "answer": "103",
    },
    {
        "id": "aime-005",
        "problem": "Compute the remainder when $3^{1000}$ is divided by $1000$.",
        "answer": "1",
    },
    {
        "id": "aime-006",
        "problem": "How many four-digit numbers have the property that the digits sum to 10?",
        "answer": "219",
    },
    {
        "id": "aime-007",
        "problem": "Find the number of ordered triples $(a, b, c)$ of positive integers satisfying $a + b + c = 20$ and $a \\leq b \\leq c$.",
        "answer": "33",
    },
    {
        "id": "aime-008",
        "problem": "A fair coin is tossed 10 times. What is the probability that no two consecutive tosses are both heads? If the probability is $\\frac{m}{n}$ in lowest terms, find $m + n$.",
        "answer": "73",
    },
    {
        "id": "aime-009",
        "problem": "Let $f(n) = n^2 + n + 1$. Find the remainder when $f(1) \\cdot f(2) \\cdot f(3) \\cdots f(10)$ is divided by 100.",
        "answer": "10",
    },
    {
        "id": "aime-010",
        "problem": "What is the last three digits of $7^{999}$?",
        "answer": "343",
    },
]


class AIMEBenchmark(BenchmarkSuite):
    """AIME: American Invitational Mathematics Examination problems."""

    name = "AIME"
    description = "Challenging competition math problems (answers are integers 000–999)"
    category = "math"

    def load_dataset(self) -> list[dict[str, Any]]:
        # Prefer aime_2026.jsonl for AIME 2026 specific problems
        for fname in ("aime_2026.jsonl", "aime.jsonl"):
            jsonl_path = Path(self.data_dir) / fname
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
            print("  No AIME data file found — using built-in 10-problem subset")
        return list(_BUILTIN_PROBLEMS)

    def build_prompt(self, problem: dict[str, Any]) -> str:
        prob = problem["problem"]
        return (
            f"Solve the following AIME-style competition math problem.\n\n"
            f"Problem: {prob}\n\n"
            f"The answer is an integer between 0 and 999. "
            f"Write ONLY the integer answer to a file called answer.txt — "
            f"no explanation, no work, just the number."
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
        numbers = re.findall(r"-?\d+", agent_output.replace(",", ""))
        if numbers:
            answer_path.write_text(numbers[-1] + "\n", encoding="utf-8")
            metadata["recovered_answer_from_output"] = True

    def evaluate(self, problem: dict[str, Any], workspace: str) -> BenchmarkResult:
        pid = problem.get("id", "unknown")
        expected = str(problem["answer"]).strip()
        answer_file = os.path.join(workspace, "answer.txt")

        if not os.path.exists(answer_file):
            return BenchmarkResult(
                problem_id=pid, passed=False, expected=expected,
                error="answer.txt not found",
            )

        with open(answer_file) as fh:
            actual_raw = fh.read().strip()

        # Extract last integer from the answer
        numbers = re.findall(r"-?\d+", actual_raw.replace(",", ""))
        actual = numbers[-1] if numbers else actual_raw

        try:
            passed = int(actual) == int(expected)
        except (ValueError, TypeError):
            passed = actual == expected

        return BenchmarkResult(
            problem_id=pid, passed=passed,
            expected=expected, actual=actual_raw,
            error="" if passed else f"expected={expected}, got={actual_raw}",
        )
