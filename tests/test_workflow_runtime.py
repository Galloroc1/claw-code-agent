from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.tools.agent_tools import build_tool_context, default_tool_registry, execute_tool
from src.core.agent_types import AgentRuntimeConfig
from src.workflow_runtime import WorkflowRuntime


class WorkflowRuntimeTests(unittest.TestCase):
    def test_workflow_runtime_discovers_and_runs_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / '.claw-workflows.json').write_text(
                (
                    '{"workflows":['
                    '{"name":"review","description":"Review the current patch.",'
                    '"steps":[{"title":"Inspect diff","detail":"Read {path}"},{"title":"Summarize"}],'
                    '"prompt":"Review changes under {path}"}'
                    ']}'
                ),
                encoding='utf-8',
            )
            runtime = WorkflowRuntime.from_workspace(workspace)
            rendered = runtime.render_workflow('review')
            run_report = runtime.render_run_report('review', arguments={'path': 'src/'})

        self.assertIn('Review the current patch', rendered)
        self.assertIn('Read src/', run_report)
        self.assertIn('Review changes under src/', run_report)

    def test_workflow_tools_execute_against_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / '.claw-workflows.json').write_text(
                (
                    '{"workflows":['
                    '{"name":"build","description":"Build the project.",'
                    '"steps":["Inspect package","Run build"]}'
                    ']}'
                ),
                encoding='utf-8',
            )
            runtime = WorkflowRuntime.from_workspace(workspace)
            context = build_tool_context(
                AgentRuntimeConfig(cwd=workspace),
                workflow_runtime=runtime,
            )
            list_result = execute_tool(
                default_tool_registry(),
                'workflow_list',
                {},
                context,
            )
            get_result = execute_tool(
                default_tool_registry(),
                'workflow_get',
                {'workflow_name': 'build'},
                context,
            )
            run_result = execute_tool(
                default_tool_registry(),
                'workflow_run',
                {'workflow_name': 'build', 'arguments': {'target': 'dist'}},
                context,
            )

        self.assertTrue(list_result.ok)
        self.assertIn('build', list_result.content)
        self.assertTrue(get_result.ok)
        self.assertIn('Build the project', get_result.content)
        self.assertTrue(run_result.ok)
        self.assertEqual(run_result.metadata.get('action'), 'workflow_run')
        self.assertIn('# Workflow Run', run_result.content)

