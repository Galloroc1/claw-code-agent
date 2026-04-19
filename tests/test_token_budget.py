from __future__ import annotations

import unittest
from unittest.mock import patch

from src.core.agent_session import AgentSessionState
from src.core.agent_context_usage import ContextUsageReport, MessageBreakdown
from src.core.agent_types import BudgetConfig
from src.token_budget import calculate_token_budget, format_token_budget


class TokenBudgetTests(unittest.TestCase):
    def test_calculate_token_budget_reports_soft_and_hard_limits(self) -> None:
        session = AgentSessionState.create(
            ['# System\nYou are helpful.'],
            'Inspect the repository and summarize the current implementation status.',
            user_context={'currentDate': "Today's date is 2026-04-11."},
            system_context={'gitStatus': 'Current branch: main'},
        )
        session.append_assistant('Reading files and checking runtime state.')

        fake_usage = ContextUsageReport(
            model='test-model',
            total_tokens=200,
            raw_max_tokens=128_000,
            percentage=0.15,
            strategy='token_budget',
            message_count=len(session.messages),
            categories=(),
            system_prompt_sections=(),
            user_context_entries=(),
            system_context_entries=(),
            memory_files=(),
            message_breakdown=MessageBreakdown(
                user_message_tokens=50,
                assistant_message_tokens=50,
                tool_call_tokens=0,
                tool_result_tokens=0,
                user_context_tokens=10,
                tool_calls_by_type=(),
            ),
            token_counter_backend='heuristic',
            token_counter_source='test',
            token_counter_accurate=False,
        )
        with patch('src.token_budget.collect_context_usage', return_value=fake_usage):
            snapshot = calculate_token_budget(
                session=session,
                model='test-model',
                budget_config=BudgetConfig(),
            )
        rendered = format_token_budget(snapshot)

        self.assertGreater(snapshot.projected_input_tokens, 0)
        self.assertGreater(snapshot.hard_input_limit_tokens, snapshot.soft_input_limit_tokens)
        self.assertGreater(snapshot.chat_overhead_tokens, 0)
        self.assertIn('# Token Budget', rendered)
        self.assertIn('Hard input limit', rendered)
        self.assertIn('Auto-compact buffer', rendered)

    def test_calculate_token_budget_honors_explicit_max_input_tokens(self) -> None:
        session = AgentSessionState.create(
            ['# System\nYou are helpful.'],
            'This prompt is deliberately longer than the tiny configured input budget. ' * 4,
        )

        fake_usage = ContextUsageReport(
            model='test-model',
            total_tokens=120,
            raw_max_tokens=128_000,
            percentage=0.09,
            strategy='token_budget',
            message_count=len(session.messages),
            categories=(),
            system_prompt_sections=(),
            user_context_entries=(),
            system_context_entries=(),
            memory_files=(),
            message_breakdown=MessageBreakdown(
                user_message_tokens=80,
                assistant_message_tokens=0,
                tool_call_tokens=0,
                tool_result_tokens=0,
                user_context_tokens=0,
                tool_calls_by_type=(),
            ),
            token_counter_backend='heuristic',
            token_counter_source='test',
            token_counter_accurate=False,
        )
        with patch('src.token_budget.collect_context_usage', return_value=fake_usage):
            snapshot = calculate_token_budget(
                session=session,
                model='test-model',
                budget_config=BudgetConfig(max_input_tokens=20),
            )

        self.assertTrue(snapshot.exceeds_hard_limit)
        self.assertGreater(snapshot.overflow_tokens, 0)
        self.assertLessEqual(snapshot.hard_input_limit_tokens, 20)
