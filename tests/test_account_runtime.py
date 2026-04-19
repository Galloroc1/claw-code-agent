from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.account_runtime import AccountRuntime
from src.tools.agent_tools import build_tool_context, default_tool_registry, execute_tool
from src.core.agent_types import AgentRuntimeConfig


class AccountRuntimeTests(unittest.TestCase):
    def test_account_runtime_discovers_profiles_and_persists_login(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / '.claw-account.json').write_text(
                (
                    '{"profiles":['
                    '{"name":"local","provider":"openai","identity":"dev@example.com","authMode":"api_key"},'
                    '{"name":"team","provider":"anthropic","identity":"team@example.com","org":"Harness"}'
                    ']}'
                ),
                encoding='utf-8',
            )
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'local-token'}, clear=False):
                runtime = AccountRuntime.from_workspace(workspace)
                report = runtime.login('local')
                restored = AccountRuntime.from_workspace(workspace)

        self.assertEqual(len(runtime.profiles), 2)
        self.assertTrue(report.logged_in)
        self.assertEqual(report.profile_name, 'local')
        self.assertIsNotNone(restored.active_session)
        self.assertEqual(restored.active_session.profile_name, 'local')
        self.assertIn('Credential env vars: OPENAI_API_KEY', restored.render_summary())

    def test_account_runtime_logout_clears_active_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / '.claw-account.json').write_text(
                '{"profiles":[{"name":"local","provider":"openai","identity":"dev@example.com"}]}',
                encoding='utf-8',
            )
            runtime = AccountRuntime.from_workspace(workspace)
            runtime.login('local')
            report = runtime.logout()

        self.assertFalse(report.logged_in)
        self.assertIn('Logged out dev@example.com', report.detail)

    def test_account_tools_execute_against_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / '.claw-account.json').write_text(
                '{"profiles":[{"name":"local","provider":"openai","identity":"dev@example.com"}]}',
                encoding='utf-8',
            )
            runtime = AccountRuntime.from_workspace(workspace)
            context = build_tool_context(
                AgentRuntimeConfig(cwd=workspace),
                account_runtime=runtime,
            )
            list_result = execute_tool(
                default_tool_registry(),
                'account_list_profiles',
                {},
                context,
            )
            login_result = execute_tool(
                default_tool_registry(),
                'account_login',
                {'target': 'local'},
                context,
            )
            status_result = execute_tool(
                default_tool_registry(),
                'account_status',
                {},
                context,
            )

        self.assertTrue(list_result.ok)
        self.assertIn('dev@example.com', list_result.content)
        self.assertTrue(login_result.ok)
        self.assertIn('profile=local', login_result.content)
        self.assertTrue(status_result.ok)
        self.assertIn('Configured account profiles: 1', status_result.content)
