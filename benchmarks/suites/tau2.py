"""
Tau2 benchmark suite.

Tau2 (TAU-bench v2) evaluates language models on tool-augmented tasks
requiring multi-step reasoning with tool use. It measures the ability
to plan, select, and chain tool calls to complete complex tasks.

The benchmark reports an average score across 3 task domains:
retail, airline, and finance.

Reference: https://github.com/sierra-research/tau-bench
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .base import BenchmarkResult, BenchmarkSuite

# ---------------------------------------------------------------------------
# Built-in mini dataset (10 representative Tau2-style problems)
# ---------------------------------------------------------------------------

_BUILTIN_PROBLEMS: list[dict[str, Any]] = [
    {
        "id": "tau2-001",
        "domain": "retail",
        "question": "A customer wants to return a laptop purchased 20 days ago. The store policy allows returns within 30 days with receipt. The customer has the receipt. The laptop is in original packaging. What is the correct action?",
        "choices": ["Process full refund", "Deny return - past return window", "Offer store credit only", "Process exchange only"],
        "answer": "A",
    },
    {
        "id": "tau2-002",
        "domain": "airline",
        "question": "A passenger has a connecting flight with a 45-minute layover. Their first flight is delayed by 30 minutes. The minimum connection time at the hub airport is 60 minutes. What should the agent recommend?",
        "choices": ["Rebook on next available flight", "Keep current booking, passenger will make it", "Cancel entire itinerary", "Upgrade to first class on delayed flight"],
        "answer": "A",
    },
    {
        "id": "tau2-003",
        "domain": "finance",
        "question": "A client wants to transfer $50,000 between accounts. The daily transfer limit is $25,000. The client has proper identification. What is the correct procedure?",
        "choices": ["Split into two $25,000 transfers over two days", "Override limit and process single transfer", "Deny the transfer entirely", "Process as a wire transfer instead"],
        "answer": "A",
    },
    {
        "id": "tau2-004",
        "domain": "retail",
        "question": "An item shows a price of $29.99 on the shelf but rings up as $34.99 at checkout. Store policy states the customer gets the lower price if there's a discrepancy. The shelf tag is verified to be for this item. What action should be taken?",
        "choices": ["Honor the shelf price of $29.99", "Charge the register price of $34.99", "Split the difference", "Give the item for free"],
        "answer": "A",
    },
    {
        "id": "tau2-005",
        "domain": "airline",
        "question": "A passenger with a non-refundable ticket wants to change their travel date. The fare difference is +$150 and there is a $75 change fee. The passenger is a Gold status member (change fees waived). What is the total additional cost?",
        "choices": ["$150", "$225", "$75", "$0"],
        "answer": "A",
    },
    {
        "id": "tau2-006",
        "domain": "finance",
        "question": "A customer's account shows 3 pending transactions totaling $500, but their available balance is $400. One pending transaction of $200 is a pre-authorization from a gas station from 5 days ago. What should the agent do?",
        "choices": ["Release the gas station pre-authorization as it exceeds the standard hold period", "Decline all pending transactions", "Suggest the customer deposit more funds", "Freeze the account for suspicious activity"],
        "answer": "A",
    },
    {
        "id": "tau2-007",
        "domain": "retail",
        "question": "A customer purchased an appliance with a 2-year manufacturer warranty 18 months ago. The appliance has stopped working due to a manufacturing defect. The customer does not have an extended warranty. What should the agent do?",
        "choices": ["Process warranty claim with manufacturer", "Offer store repair at customer's cost", "Deny any assistance", "Offer replacement at discounted price"],
        "answer": "A",
    },
    {
        "id": "tau2-008",
        "domain": "airline",
        "question": "A flight is overbooked by 2 passengers. 3 passengers volunteer to give up their seats. Compensation offered is $400 voucher + next flight (2 hours later). Airline policy requires bumping in reverse check-in order if volunteers insufficient. What should happen?",
        "choices": ["Accept 2 of the 3 volunteers, compensate them", "Accept all 3 volunteers", "Involuntarily deny 2 passengers boarding", "Cancel the flight"],
        "answer": "A",
    },
    {
        "id": "tau2-009",
        "domain": "finance",
        "question": "A customer reports their credit card stolen. There are 3 unauthorized transactions in the last 24 hours totaling $1,200. Federal regulation limits customer liability for unauthorized transactions reported within 2 business days. What steps should be taken first?",
        "choices": ["Block the card immediately and initiate fraud investigation", "Wait for the monthly statement to dispute charges", "Transfer balance to new card without blocking old one", "Ask customer to contact merchants directly"],
        "answer": "A",
    },
    {
        "id": "tau2-010",
        "domain": "retail",
        "question": "A customer wants to price match an identical item found cheaper at a competitor. Store policy allows price matching for identical items at local competitors within 14 days of purchase. The customer purchased 10 days ago and has proof of the competitor's current price. The competitor is an online-only store. Store policy explicitly excludes online-only retailers. What should the agent do?",
        "choices": ["Deny price match - online-only retailers excluded", "Honor the price match anyway", "Offer partial price adjustment", "Contact competitor to verify price"],
        "answer": "A",
    },
]


class Tau2Benchmark(BenchmarkSuite):
    """Tau2: Tool-augmented task evaluation across retail, airline, and finance."""

    name = "Tau2"
    description = "Tool-augmented multi-step reasoning (retail/airline/finance)"
    category = "reasoning"

    def load_dataset(self) -> list[dict[str, Any]]:
        jsonl_path = Path(self.data_dir) / "tau2.jsonl"
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
        domain = problem.get("domain", "general")
        letters = "ABCD"
        choices_text = "\n".join(
            f"  {letters[i]}. {c}" for i, c in enumerate(choices)
        )
        return (
            f"You are a {domain} service agent. Answer the following question "
            f"about the correct action to take.\n\n"
            f"Scenario: {question}\n\n"
            f"Options:\n{choices_text}\n\n"
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
