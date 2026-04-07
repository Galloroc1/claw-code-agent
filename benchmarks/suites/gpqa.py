"""
GPQA Diamond benchmark suite.

GPQA (Graduate-Level Google-Proof Question Answering) Diamond is a subset
of extremely difficult questions written by domain experts in biology,
physics, and chemistry. The "Diamond" subset is the hardest tier.

Reference: https://huggingface.co/datasets/Idavidrein/gpqa
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .base import BenchmarkResult, BenchmarkSuite

# ---------------------------------------------------------------------------
# Built-in mini dataset (10 representative GPQA Diamond problems)
# ---------------------------------------------------------------------------

_BUILTIN_PROBLEMS: list[dict[str, Any]] = [
    {
        "id": "gpqa-001",
        "subject": "physics",
        "question": "A particle of mass m is confined to a one-dimensional box of length L. What is the energy difference between the first excited state and the ground state?",
        "choices": ["3π²ℏ²/(2mL²)", "π²ℏ²/(2mL²)", "4π²ℏ²/(2mL²)", "2π²ℏ²/(mL²)"],
        "answer": "A",
    },
    {
        "id": "gpqa-002",
        "subject": "chemistry",
        "question": "Which of the following molecules has the highest bond dissociation energy?",
        "choices": ["N₂", "O₂", "F₂", "CO"],
        "answer": "D",
    },
    {
        "id": "gpqa-003",
        "subject": "biology",
        "question": "In the lac operon, which component acts as the inducer that causes the repressor to release from the operator?",
        "choices": ["Lactose", "Allolactose", "Glucose", "cAMP"],
        "answer": "B",
    },
    {
        "id": "gpqa-004",
        "subject": "physics",
        "question": "In quantum electrodynamics, what is the leading-order correction to the electron g-factor (anomalous magnetic moment)?",
        "choices": ["α/(2π)", "α/π", "α²/(2π)", "2α/π"],
        "answer": "A",
    },
    {
        "id": "gpqa-005",
        "subject": "chemistry",
        "question": "What is the primary product when 2-methylpropene undergoes hydroboration-oxidation?",
        "choices": ["2-methylpropan-2-ol", "2-methylpropan-1-ol", "2-methylpropanal", "isobutylene oxide"],
        "answer": "B",
    },
    {
        "id": "gpqa-006",
        "subject": "biology",
        "question": "Which of the following enzymes is responsible for adding a 5' cap to mRNA in eukaryotes?",
        "choices": ["RNA polymerase II", "Guanylyltransferase", "Poly(A) polymerase", "RNA triphosphatase alone"],
        "answer": "B",
    },
    {
        "id": "gpqa-007",
        "subject": "physics",
        "question": "In general relativity, the Schwarzschild radius of a black hole with mass M is given by:",
        "choices": ["2GM/c²", "GM/c²", "GM²/c", "2GM²/c²"],
        "answer": "A",
    },
    {
        "id": "gpqa-008",
        "subject": "chemistry",
        "question": "Which of the following is the correct order of acidity for the hydrogen halides in aqueous solution?",
        "choices": ["HF > HCl > HBr > HI", "HI > HBr > HCl > HF", "HCl > HBr > HI > HF", "HF > HI > HBr > HCl"],
        "answer": "B",
    },
    {
        "id": "gpqa-009",
        "subject": "biology",
        "question": "What is the primary function of topoisomerase II during DNA replication?",
        "choices": ["Unwinding the double helix", "Relieving positive supercoiling ahead of the replication fork", "Joining Okazaki fragments", "Proofreading newly synthesized DNA"],
        "answer": "B",
    },
    {
        "id": "gpqa-010",
        "subject": "physics",
        "question": "In the Standard Model, the Higgs mechanism gives mass to which fundamental particles?",
        "choices": ["Only quarks", "Only W and Z bosons", "W bosons, Z bosons, and all fermions", "All particles including photons and gluons"],
        "answer": "C",
    },
]


class GPQABenchmark(BenchmarkSuite):
    """GPQA Diamond: Graduate-level science QA (physics, chemistry, biology)."""

    name = "GPQA-Diamond"
    description = "Graduate-level science multiple-choice (diamond difficulty)"
    category = "knowledge"

    def load_dataset(self) -> list[dict[str, Any]]:
        jsonl_path = Path(self.data_dir) / "gpqa.jsonl"
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
        letters = "ABCD"
        choices_text = "\n".join(
            f"  {letters[i]}. {c}" for i, c in enumerate(choices)
        )
        return (
            f"Answer the following graduate-level science question.\n\n"
            f"Question: {question}\n\n"
            f"Choices:\n{choices_text}\n\n"
            f"Think carefully and write ONLY the letter of the correct answer (A–D) "
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
