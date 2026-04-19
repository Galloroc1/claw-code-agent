from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.tools.agent_tools import build_tool_context, default_tool_registry, execute_tool
from src.core.agent_types import AgentPermissions, AgentRuntimeConfig
from src.config_runtime import ConfigRuntime


class ConfigRuntimeTests(unittest.TestCase):
    def test_config_runtime_loads_and_merges_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            claude_dir = workspace / '.claude'
            claude_dir.mkdir()
            (claude_dir / 'settings.json').write_text(
                '{"model":{"name":"project-model","temperature":0.1},"review":{"strict":false}}',
                encoding='utf-8',
            )
            (claude_dir / 'settings.local.json').write_text(
                '{"model":{"temperature":0.0},"review":{"strict":true}}',
                encoding='utf-8',
            )
            runtime = ConfigRuntime.from_workspace(workspace)

        self.assertTrue(runtime.has_config())
        self.assertEqual(runtime.get_value('model.name'), 'project-model')
        self.assertEqual(runtime.get_value('model.temperature'), 0.0)
        self.assertEqual(runtime.get_value('review.strict'), True)
        self.assertIn('Config sources: 2', runtime.render_summary())

    def test_config_runtime_persists_set_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            runtime = ConfigRuntime.from_workspace(workspace)
            mutation = runtime.set_value('review.mode', 'strict', source='local')
            restored = ConfigRuntime.from_workspace(workspace)

        self.assertEqual(mutation.source_name, 'local')
        self.assertEqual(restored.get_value('review.mode'), 'strict')
        self.assertIn('review.mode', restored.render_keys())
        self.assertEqual(json.loads(restored.render_value('review.mode')), 'strict')

    def test_config_tools_execute_against_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            runtime = ConfigRuntime.from_workspace(workspace)
            context = build_tool_context(
                AgentRuntimeConfig(
                    cwd=workspace,
                    permissions=AgentPermissions(allow_file_write=True),
                ),
                config_runtime=runtime,
            )
            set_result = execute_tool(
                default_tool_registry(),
                'config_set',
                {'key_path': 'review.mode', 'value': 'strict'},
                context,
            )
            list_result = execute_tool(
                default_tool_registry(),
                'config_list',
                {'prefix': 'review'},
                context,
            )
            get_result = execute_tool(
                default_tool_registry(),
                'config_get',
                {'key_path': 'review.mode'},
                context,
            )

        self.assertTrue(set_result.ok)
        self.assertEqual(set_result.metadata.get('source_name'), 'local')
        self.assertIn('# Config Keys', list_result.content)
        self.assertIn('review.mode', list_result.content)
        self.assertIn('# Config Value', get_result.content)
        self.assertIn('"strict"', get_result.content)
