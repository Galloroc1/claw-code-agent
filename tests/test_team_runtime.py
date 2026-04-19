from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.tools.agent_tools import build_tool_context, default_tool_registry, execute_tool
from src.core.agent_types import AgentPermissions, AgentRuntimeConfig
from src.team_runtime import TeamRuntime


class TeamRuntimeTests(unittest.TestCase):
    def test_team_runtime_persists_team_and_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            runtime = TeamRuntime.from_workspace(workspace)
            runtime.create_team('reviewers', members=['alice', 'bob'])
            runtime.send_message(
                team_name='reviewers',
                text='Please review the patch.',
                sender='agent',
                recipient='alice',
            )
            restored = TeamRuntime.from_workspace(workspace)

        self.assertEqual(len(restored.teams), 1)
        self.assertEqual(restored.teams[0].name, 'reviewers')
        self.assertEqual(len(restored.messages), 1)
        self.assertIn('Please review the patch.', restored.render_messages())

    def test_team_tools_execute_against_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            runtime = TeamRuntime.from_workspace(workspace)
            context = build_tool_context(
                AgentRuntimeConfig(
                    cwd=workspace,
                    permissions=AgentPermissions(allow_file_write=True),
                ),
                team_runtime=runtime,
            )
            create_result = execute_tool(
                default_tool_registry(),
                'team_create',
                {'team_name': 'reviewers', 'members': ['alice', 'bob']},
                context,
            )
            send_result = execute_tool(
                default_tool_registry(),
                'send_message',
                {
                    'team_name': 'reviewers',
                    'message': 'Check notebook changes',
                    'sender': 'agent',
                },
                context,
            )
            list_result = execute_tool(
                default_tool_registry(),
                'team_list',
                {},
                context,
            )
            message_result = execute_tool(
                default_tool_registry(),
                'team_messages',
                {'team_name': 'reviewers'},
                context,
            )

        self.assertTrue(create_result.ok)
        self.assertEqual(create_result.metadata.get('action'), 'team_create')
        self.assertTrue(send_result.ok)
        self.assertEqual(send_result.metadata.get('action'), 'send_message')
        self.assertIn('reviewers', list_result.content)
        self.assertIn('Check notebook changes', message_result.content)
