from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.ask_user_runtime import AskUserRuntime
from src.tools.agent_tools import build_tool_context, default_tool_registry, execute_tool
from src.core.agent_types import AgentRuntimeConfig


class AskUserRuntimeTests(unittest.TestCase):
    def test_ask_user_runtime_consumes_queued_answers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / '.claw-ask-user.json').write_text(
                '{"answers":[{"question":"Approve deploy?","answer":"yes"}]}',
                encoding='utf-8',
            )
            runtime = AskUserRuntime.from_workspace(workspace)
            response = runtime.answer(question='Approve deploy?')
            restored = AskUserRuntime.from_workspace(workspace)

        self.assertEqual(response.answer, 'yes')
        self.assertEqual(response.source, 'queued')
        self.assertEqual(len(restored.queued_answers), 0)
        self.assertEqual(len(restored.history), 1)

    def test_ask_user_tool_executes_against_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / '.claw-ask-user.json').write_text(
                '{"answers":[{"question":"Choose mode","answer":"safe"}]}',
                encoding='utf-8',
            )
            runtime = AskUserRuntime.from_workspace(workspace)
            context = build_tool_context(
                AgentRuntimeConfig(cwd=workspace),
                ask_user_runtime=runtime,
            )
            result = execute_tool(
                default_tool_registry(),
                'ask_user_question',
                {
                    'question': 'Choose mode',
                    'choices': ['safe', 'fast'],
                    'allow_free_text': False,
                },
                context,
            )

        self.assertTrue(result.ok)
        self.assertIn('# Ask User', result.content)
        self.assertIn('safe', result.content)
        self.assertEqual(result.metadata.get('action'), 'ask_user_question')
