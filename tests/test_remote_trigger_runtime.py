from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.tools.agent_tools import build_tool_context, default_tool_registry, execute_tool
from src.core.agent_types import AgentPermissions, AgentRuntimeConfig
from src.remote_trigger_runtime import RemoteTriggerRuntime


class RemoteTriggerRuntimeTests(unittest.TestCase):
    def test_remote_trigger_runtime_discovers_and_runs_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / '.claw-triggers.json').write_text(
                (
                    '{"triggers":['
                    '{"trigger_id":"nightly","name":"Nightly","workflow":"review",'
                    '"schedule":"0 0 * * *","body":{"depth":"full"}}'
                    ']}'
                ),
                encoding='utf-8',
            )
            runtime = RemoteTriggerRuntime.from_workspace(workspace)
            trigger_report = runtime.render_trigger('nightly')
            run_report = runtime.render_run_report('nightly', body={'depth': 'quick'})

        self.assertIn('trigger_id=nightly', trigger_report)
        self.assertIn('workflow=review', run_report)
        self.assertIn('"depth": "quick"', run_report)

    def test_remote_trigger_tool_supports_create_update_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            runtime = RemoteTriggerRuntime.from_workspace(workspace)
            context = build_tool_context(
                AgentRuntimeConfig(
                    cwd=workspace,
                    permissions=AgentPermissions(allow_file_write=True),
                ),
                remote_trigger_runtime=runtime,
            )
            create_result = execute_tool(
                default_tool_registry(),
                'remote_trigger',
                {
                    'action': 'create',
                    'body': {
                        'trigger_id': 'nightly',
                        'name': 'Nightly',
                        'workflow': 'review',
                        'body': {'depth': 'full'},
                    },
                },
                context,
            )
            update_result = execute_tool(
                default_tool_registry(),
                'remote_trigger',
                {
                    'action': 'update',
                    'trigger_id': 'nightly',
                    'body': {'schedule': '0 0 * * *'},
                },
                context,
            )
            run_result = execute_tool(
                default_tool_registry(),
                'remote_trigger',
                {
                    'action': 'run',
                    'trigger_id': 'nightly',
                    'body': {'depth': 'quick'},
                },
                context,
            )

        self.assertTrue(create_result.ok)
        self.assertEqual(create_result.metadata.get('trigger_id'), 'nightly')
        self.assertTrue(update_result.ok)
        self.assertEqual(update_result.metadata.get('remote_trigger_action'), 'update')
        self.assertTrue(run_result.ok)
        self.assertIn('# Remote Trigger Run', run_result.content)
