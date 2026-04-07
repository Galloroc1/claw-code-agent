"""
MMLU-Pro benchmark suite.

MMLU-Pro (Massive Multitask Language Understanding - Professional) is an
enhanced version of MMLU with harder questions, 10 answer choices (A–J),
and chain-of-thought reasoning requirements across 14 subjects.

Reference: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .base import BenchmarkResult, BenchmarkSuite

# ---------------------------------------------------------------------------
# Built-in mini dataset (10 representative MMLU-Pro problems)
# ---------------------------------------------------------------------------

_BUILTIN_PROBLEMS: list[dict[str, Any]] = [
    {
        "id": "mmlu-pro-001",
        "subject": "computer_science",
        "question": "Which of the following best describes the time complexity of binary search on a sorted array of n elements?",
        "choices": ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(2^n)", "O(n!)", "O(sqrt(n))", "O(n^3)", "O(log log n)"],
        "answer": "B",
    },
    {
        "id": "mmlu-pro-002",
        "subject": "physics",
        "question": "A 2 kg block is placed on a frictionless inclined plane that makes a 30° angle with the horizontal. What is the acceleration of the block down the incline? (g = 9.8 m/s²)",
        "choices": ["2.45 m/s²", "4.9 m/s²", "6.93 m/s²", "9.8 m/s²", "3.27 m/s²", "8.49 m/s²", "1.63 m/s²", "5.66 m/s²", "7.35 m/s²", "0.98 m/s²"],
        "answer": "B",
    },
    {
        "id": "mmlu-pro-003",
        "subject": "chemistry",
        "question": "What is the hybridization of the central atom in SF6?",
        "choices": ["sp", "sp2", "sp3", "sp3d", "sp3d2", "dsp3", "d2sp3", "sp2d", "sp3d3", "none of the above"],
        "answer": "E",
    },
    {
        "id": "mmlu-pro-004",
        "subject": "mathematics",
        "question": "What is the derivative of f(x) = x³ · ln(x)?",
        "choices": ["3x² · ln(x)", "x² + 3x² · ln(x)", "3x² · ln(x) + x²", "x³/x + 3x²", "3x² · ln(x) + x³", "x² · (3ln(x) + 1)", "3x · ln(x) + x²", "x³ · (1/x) + 3x² · ln(x)", "3x² / ln(x)", "ln(x) + 3x²"],
        "answer": "F",
    },
    {
        "id": "mmlu-pro-005",
        "subject": "biology",
        "question": "Which of the following is NOT a characteristic of the adaptive immune response?",
        "choices": ["Specificity", "Memory", "Rapid initial response", "Diversity", "Self/non-self discrimination", "Clonal expansion", "Tolerance", "MHC restriction", "Somatic hypermutation", "Pattern recognition"],
        "answer": "C",
    },
    {
        "id": "mmlu-pro-006",
        "subject": "history",
        "question": "The Treaty of Westphalia (1648) is most significant because it:",
        "choices": ["Ended the Hundred Years War", "Established the principle of state sovereignty", "Created the United Nations", "Divided Africa among European powers", "Ended World War I", "Established the Holy Roman Empire", "Created NATO", "Ended the Napoleonic Wars", "Established the European Union", "Ended the Crusades"],
        "answer": "B",
    },
    {
        "id": "mmlu-pro-007",
        "subject": "economics",
        "question": "In a perfectly competitive market in long-run equilibrium, which of the following is true?",
        "choices": ["Price equals minimum ATC", "Firms earn positive economic profit", "Price exceeds marginal cost", "Firms produce at maximum ATC", "Barriers to entry exist", "Price equals maximum AVC", "Firms are price makers", "Marginal cost exceeds price", "Economic profit is negative", "Supply is perfectly inelastic"],
        "answer": "A",
    },
    {
        "id": "mmlu-pro-008",
        "subject": "philosophy",
        "question": "According to Kant's categorical imperative, an action is morally permissible if and only if:",
        "choices": ["It maximizes overall happiness", "It can be universalized without contradiction", "It promotes the greatest good for the greatest number", "God commands it", "It follows natural law", "It leads to virtuous character", "It is agreed upon by rational contractors", "It satisfies one's desires", "It is consistent with social norms", "It minimizes suffering"],
        "answer": "B",
    },
    {
        "id": "mmlu-pro-009",
        "subject": "law",
        "question": "Under the U.S. Constitution, which amendment protects against unreasonable searches and seizures?",
        "choices": ["First Amendment", "Second Amendment", "Third Amendment", "Fourth Amendment", "Fifth Amendment", "Sixth Amendment", "Seventh Amendment", "Eighth Amendment", "Ninth Amendment", "Tenth Amendment"],
        "answer": "D",
    },
    {
        "id": "mmlu-pro-010",
        "subject": "psychology",
        "question": "In Piaget's theory of cognitive development, the stage in which children develop the ability to think abstractly and reason hypothetically is called the:",
        "choices": ["Sensorimotor stage", "Preoperational stage", "Concrete operational stage", "Formal operational stage", "Postformal stage", "Latency stage", "Genital stage", "Identity vs. role confusion", "Autonomy vs. shame", "Initiative vs. guilt"],
        "answer": "D",
    },
]


class MMLUProBenchmark(BenchmarkSuite):
    """MMLU-Pro: Enhanced multitask language understanding with 10-choice questions."""

    name = "MMLU-Pro"
    description = "Professional-level multiple-choice QA across 14 subjects (10 choices)"
    category = "knowledge"

    def load_dataset(self) -> list[dict[str, Any]]:
        jsonl_path = Path(self.data_dir) / "mmlu_pro.jsonl"
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
            f"Answer the following multiple-choice question.\n\n"
            f"Question: {question}\n\n"
            f"Choices:\n{choices_text}\n\n"
            f"Write ONLY the letter of the correct answer (A–J) to a file called answer.txt — "
            f"no explanation, just the single letter."
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
