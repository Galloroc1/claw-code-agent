"""
MMMLU (Multilingual Massive Multitask Language Understanding) benchmark suite.

MMMLU extends MMLU to multiple languages, testing language understanding
across diverse cultural and linguistic contexts.

Reference: https://huggingface.co/datasets/openai/MMMLU
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .base import BenchmarkResult, BenchmarkSuite

# ---------------------------------------------------------------------------
# Built-in mini dataset (10 representative MMMLU problems — multilingual)
# ---------------------------------------------------------------------------

_BUILTIN_PROBLEMS: list[dict[str, Any]] = [
    {
        "id": "mmmlu-001",
        "language": "en",
        "subject": "abstract_algebra",
        "question": "Find the degree of the extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.",
        "choices": ["0", "4", "2", "6"],
        "answer": "B",
    },
    {
        "id": "mmmlu-002",
        "language": "es",
        "subject": "anatomy",
        "question": "¿Cuál de los siguientes músculos es el principal responsable de la abducción del brazo?",
        "choices": ["Deltoides", "Trapecio", "Pectoral mayor", "Dorsal ancho"],
        "answer": "A",
    },
    {
        "id": "mmmlu-003",
        "language": "fr",
        "subject": "astronomy",
        "question": "Quelle est la planète la plus proche du Soleil?",
        "choices": ["Vénus", "Terre", "Mercure", "Mars"],
        "answer": "C",
    },
    {
        "id": "mmmlu-004",
        "language": "de",
        "subject": "business_ethics",
        "question": "Welcher der folgenden Begriffe beschreibt die Verantwortung eines Unternehmens gegenüber der Gesellschaft am besten?",
        "choices": ["Corporate Social Responsibility", "Shareholder Theory", "Laissez-faire", "Monopol"],
        "answer": "A",
    },
    {
        "id": "mmmlu-005",
        "language": "zh",
        "subject": "computer_science",
        "question": "在计算机科学中，二叉搜索树的平均时间复杂度是多少？",
        "choices": ["O(n)", "O(log n)", "O(n²)", "O(1)"],
        "answer": "B",
    },
    {
        "id": "mmmlu-006",
        "language": "ja",
        "subject": "world_history",
        "question": "フランス革命が始まったのは何年ですか？",
        "choices": ["1776年", "1789年", "1804年", "1815年"],
        "answer": "B",
    },
    {
        "id": "mmmlu-007",
        "language": "pt",
        "subject": "geography",
        "question": "Qual é o rio mais longo do mundo?",
        "choices": ["Amazonas", "Nilo", "Mississipi", "Yangtzé"],
        "answer": "B",
    },
    {
        "id": "mmmlu-008",
        "language": "it",
        "subject": "philosophy",
        "question": "Chi ha scritto 'La Repubblica'?",
        "choices": ["Aristotele", "Platone", "Socrate", "Epicuro"],
        "answer": "B",
    },
    {
        "id": "mmmlu-009",
        "language": "ko",
        "subject": "physics",
        "question": "뉴턴의 제2법칙에서 힘의 공식은?",
        "choices": ["F = mv", "F = ma", "F = mg", "F = mc²"],
        "answer": "B",
    },
    {
        "id": "mmmlu-010",
        "language": "ar",
        "subject": "chemistry",
        "question": "ما هو العنصر الأكثر وفرة في القشرة الأرضية؟",
        "choices": ["الحديد", "الأكسجين", "السيليكون", "الألمنيوم"],
        "answer": "B",
    },
]


class MMMMLUBenchmark(BenchmarkSuite):
    """MMMLU: Multilingual MMLU across diverse languages."""

    name = "MMMLU"
    description = "Multilingual multiple-choice QA across languages and subjects"
    category = "knowledge"

    def load_dataset(self) -> list[dict[str, Any]]:
        jsonl_path = Path(self.data_dir) / "mmmlu.jsonl"
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
        language = problem.get("language", "en")
        letters = "ABCD"
        choices_text = "\n".join(
            f"  {letters[i]}. {c}" for i, c in enumerate(choices)
        )

        lang_instruction = ""
        if language != "en":
            lang_instruction = f"The question is in {language}. "

        return (
            f"Answer the following multiple-choice question. {lang_instruction}\n\n"
            f"Question: {question}\n\n"
            f"Choices:\n{choices_text}\n\n"
            f"Write ONLY the letter of the correct answer (A–D) to a file called answer.txt — "
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
        match = re.search(r"\b([A-D])\b", agent_output)
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

        match = re.search(r"\b([A-D])\b", actual_raw.upper())
        actual = match.group(1) if match else actual_raw.strip().upper()

        passed = actual == expected
        return BenchmarkResult(
            problem_id=pid, passed=passed,
            expected=expected, actual=actual_raw,
            error="" if passed else f"expected={expected}, got={actual_raw}",
        )
