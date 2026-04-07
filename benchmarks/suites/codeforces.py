"""
Codeforces benchmark suite.

Evaluates competitive programming ability using Codeforces-style problems.
Unlike pass/fail suites, this suite computes an ELO-like rating based on
problem difficulty and correctness.

Reference: https://codeforces.com/
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

from .base import BenchmarkResult, BenchmarkSuite, SuiteReport

# ---------------------------------------------------------------------------
# Built-in mini dataset (10 Codeforces-style problems with difficulty ratings)
# ---------------------------------------------------------------------------

_BUILTIN_PROBLEMS: list[dict[str, Any]] = [
    {
        "id": "cf-800-001",
        "rating": 800,
        "title": "Watermelon",
        "problem": "Pete and Billy have a watermelon weighing w kilograms. They want to divide it into two parts, each weighing an even number of kilograms. Determine if this is possible.\n\nInput: A single integer w (1 ≤ w ≤ 100)\nOutput: Print YES if possible, NO otherwise.",
        "test_cases": [
            {"input": "8", "expected_output": "YES"},
            {"input": "3", "expected_output": "NO"},
            {"input": "1", "expected_output": "NO"},
            {"input": "2", "expected_output": "NO"},
            {"input": "4", "expected_output": "YES"},
        ],
    },
    {
        "id": "cf-800-002",
        "rating": 800,
        "title": "Way Too Long Words",
        "problem": "Abbreviate words longer than 10 characters. For such words, output the first letter, number of middle characters, and last letter.\n\nInput: First line is n (1 ≤ n ≤ 100). Next n lines each contain a word.\nOutput: For each word, output the abbreviation or the word itself if length ≤ 10.",
        "test_cases": [
            {"input": "4\nword\nlocalization\ninternationalization\npneumonoultramicroscopicsilicovolcanoconiosis", "expected_output": "word\nl10n\ni18n\np43s"},
            {"input": "1\nabcdefghij", "expected_output": "abcdefghij"},
        ],
    },
    {
        "id": "cf-1000-001",
        "rating": 1000,
        "title": "Nearly Lucky Number",
        "problem": "A number is nearly lucky if the count of digits 4 and 7 in it is itself a lucky number (composed only of 4s and 7s). Given n, determine if it is nearly lucky.\n\nInput: A single integer n (1 ≤ n ≤ 10^18)\nOutput: YES or NO",
        "test_cases": [
            {"input": "40047", "expected_output": "NO"},
            {"input": "7747774", "expected_output": "YES"},
            {"input": "1000000000000000000", "expected_output": "NO"},
        ],
    },
    {
        "id": "cf-1200-001",
        "rating": 1200,
        "title": "Beautiful Matrix",
        "problem": "You have a 5×5 matrix with exactly one 1 and rest 0s. In one move, you can swap any two adjacent rows or columns. Find the minimum number of moves to place the 1 in the center (row 3, col 3).\n\nInput: 5 lines each with 5 space-separated integers.\nOutput: Minimum number of moves.",
        "test_cases": [
            {"input": "0 0 0 0 0\n0 0 0 0 1\n0 0 0 0 0\n0 0 0 0 0\n0 0 0 0 0", "expected_output": "3"},
            {"input": "0 0 0 0 0\n0 0 0 0 0\n0 0 1 0 0\n0 0 0 0 0\n0 0 0 0 0", "expected_output": "0"},
        ],
    },
    {
        "id": "cf-1400-001",
        "rating": 1400,
        "title": "Kefa and Park",
        "problem": "Kefa wants to walk from the root (node 1) to any leaf in a tree. Along the path, there are cats at some nodes. Find the number of leaves reachable such that the path doesn't contain more than m consecutive cats.\n\nInput: First line: n m (n nodes, max m consecutive cats). Second line: n integers (0/1) for each node. Next n-1 lines: edges.\nOutput: Number of valid leaves.\n\nFor the built-in test, n=7 m=1, cats=[1,1,0,0,1,0,1], edges: 1-2, 1-3, 2-4, 2-5, 3-6, 3-7",
        "test_cases": [
            {"input": "7 1\n1 1 0 0 1 0 1\n1 2\n1 3\n2 4\n2 5\n3 6\n3 7", "expected_output": "2"},
        ],
    },
    {
        "id": "cf-1600-001",
        "rating": 1600,
        "title": "Divisibility by Eight",
        "problem": "Given a number (as a string of up to 100 digits), determine if you can delete some digits to get a non-empty number divisible by 8. Leading zeros are allowed in the result.\n\nInput: A string of digits.\nOutput: YES and the resulting number, or NO.",
        "test_cases": [
            {"input": "3121", "expected_output": "YES\n312"},
            {"input": "123456789", "expected_output": "YES\n8"},
            {"input": "3", "expected_output": "NO"},
        ],
    },
    {
        "id": "cf-1800-001",
        "rating": 1800,
        "title": "Array Partition",
        "problem": "Given an array a of n integers, determine if you can partition it into three non-empty contiguous parts such that max(part1) = min(part2) = max(part3).\n\nInput: First line n. Second line: a1...an.\nOutput: YES and the lengths of three parts, or NO.",
        "test_cases": [
            {"input": "5\n3 1 5 3 1", "expected_output": "YES"},
            {"input": "3\n1 2 3", "expected_output": "NO"},
        ],
    },
    {
        "id": "cf-2000-001",
        "rating": 2000,
        "title": "Count Binary Strings",
        "problem": "Count the number of binary strings of length n where no two adjacent characters are both '1'. Output the answer modulo 10^9 + 7.\n\nInput: A single integer n (1 ≤ n ≤ 10^6)\nOutput: The count mod 10^9+7.",
        "test_cases": [
            {"input": "1", "expected_output": "2"},
            {"input": "2", "expected_output": "3"},
            {"input": "3", "expected_output": "5"},
            {"input": "10", "expected_output": "144"},
        ],
    },
    {
        "id": "cf-2200-001",
        "rating": 2200,
        "title": "Xor Sequences",
        "problem": "Given an array of n distinct non-negative integers, find the number of pairs (i,j) where i < j such that a[i] XOR a[j] has an even number of set bits.\n\nInput: First line n. Second line: a1...an.\nOutput: Number of such pairs.",
        "test_cases": [
            {"input": "3\n1 2 3", "expected_output": "1"},
            {"input": "4\n0 1 2 3", "expected_output": "2"},
        ],
    },
    {
        "id": "cf-2500-001",
        "rating": 2500,
        "title": "Segment Tree Query",
        "problem": "Given an array of n integers, answer q queries. Each query gives l, r and asks for the sum of elements from index l to r (1-indexed). Implement this efficiently.\n\nInput: First line: n q. Second line: a1...an. Next q lines: l r.\nOutput: q lines with answers.",
        "test_cases": [
            {"input": "5 3\n1 2 3 4 5\n1 3\n2 4\n1 5", "expected_output": "6\n9\n15"},
        ],
    },
]


def _compute_elo(results: list[BenchmarkResult], problems: list[dict[str, Any]]) -> float:
    """Compute an approximate ELO rating from problem results and difficulty ratings."""
    if not results:
        return 0.0

    # Build a mapping from problem_id to rating
    rating_map = {str(p.get("id", "")): p.get("rating", 1200) for p in problems}

    # Start with base rating 800
    elo = 800.0
    k_factor = 40.0

    for result in results:
        problem_rating = rating_map.get(result.problem_id, 1200)
        # Expected score based on ELO difference
        expected = 1.0 / (1.0 + math.pow(10, (problem_rating - elo) / 400.0))
        actual_score = 1.0 if result.passed else 0.0
        elo += k_factor * (actual_score - expected)

    return round(max(0, elo))


class CodeforcesBenchmark(BenchmarkSuite):
    """Codeforces: Competitive programming with ELO-based scoring."""

    name = "Codeforces"
    description = "Competitive programming problems with ELO rating estimation"
    category = "coding"

    def load_dataset(self) -> list[dict[str, Any]]:
        jsonl_path = Path(self.data_dir) / "codeforces.jsonl"
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
        title = problem.get("title", "Problem")
        text = problem["problem"]
        test_cases = problem.get("test_cases", [])

        examples = ""
        for i, tc in enumerate(test_cases[:2], 1):
            examples += f"\nExample {i}:\n  Input: {tc['input']}\n  Output: {tc['expected_output']}\n"

        return (
            f"Solve the following competitive programming problem.\n\n"
            f"Title: {title}\n\n"
            f"Problem:\n{text}\n"
            f"{examples}\n"
            f"Write a Python solution in a file called solution.py that reads from "
            f"stdin and writes to stdout."
        )

    def setup_workspace(self, problem: dict[str, Any], workspace: str) -> None:
        # Write test cases to workspace for evaluation
        test_cases = problem.get("test_cases", [])
        for i, tc in enumerate(test_cases):
            input_file = os.path.join(workspace, f"test_input_{i}.txt")
            expected_file = os.path.join(workspace, f"test_expected_{i}.txt")
            with open(input_file, "w") as f:
                f.write(tc["input"] + "\n")
            with open(expected_file, "w") as f:
                f.write(tc["expected_output"].strip() + "\n")

    def evaluate(self, problem: dict[str, Any], workspace: str) -> BenchmarkResult:
        pid = problem.get("id", "unknown")
        rating = problem.get("rating", 1200)
        solution_file = os.path.join(workspace, "solution.py")

        if not os.path.exists(solution_file):
            return BenchmarkResult(
                problem_id=pid, passed=False,
                error="solution.py not found",
                metadata={"rating": rating},
            )

        test_cases = problem.get("test_cases", [])
        all_passed = True
        errors = []

        for i, tc in enumerate(test_cases):
            input_data = tc["input"]
            expected = tc["expected_output"].strip()

            rc, output = self._run_shell(
                f"echo {repr(input_data)} | python3 solution.py",
                cwd=workspace,
                timeout=10.0,
            )

            actual = output.strip()
            if rc != 0:
                all_passed = False
                errors.append(f"test {i}: runtime error: {output[:200]}")
            elif actual != expected:
                # Check if the output matches any valid answer
                # (for problems with YES/NO + additional output)
                if expected.startswith("YES") and actual.startswith("YES"):
                    pass  # Accept if at least "YES" is correct
                elif expected.startswith("NO") and actual.startswith("NO"):
                    pass
                else:
                    all_passed = False
                    errors.append(f"test {i}: expected={expected!r}, got={actual!r}")

        return BenchmarkResult(
            problem_id=pid, passed=all_passed,
            expected=f"all {len(test_cases)} tests",
            actual=f"{'all passed' if all_passed else '; '.join(errors)}",
            error="" if all_passed else "; ".join(errors),
            metadata={"rating": rating},
        )

    def run_all(self) -> SuiteReport:
        """Override to add ELO computation to the report."""
        report = super().run_all()
        problems = self.load_dataset()
        if self.limit is not None:
            problems = problems[: self.limit]
        elo = _compute_elo(report.results, problems)
        print(f"  Estimated Codeforces ELO: {elo}")
        report.results.append(
            BenchmarkResult(
                problem_id="_elo_rating",
                passed=True,
                actual=str(elo),
                metadata={"elo_rating": elo},
            )
        )
        return report
