from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.tools.agent_tools import build_tool_context, default_tool_registry, execute_tool
from src.core.agent_types import AgentRuntimeConfig
from src.remote_runtime import (
    RemoteRuntime,
    run_deep_link_mode,
    run_direct_connect_mode,
    run_remote_mode,
    run_ssh_mode,
    run_teleport_mode,
)


class RemoteRuntimeTests(unittest.TestCase):
    def test_remote_runtime_discovers_profiles_and_persists_connection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / '.claw-remote.json').write_text(
                (
                    '{"profiles":['
                    '{"name":"staging","mode":"ssh","target":"dev@staging","workspaceCwd":"/srv/app"},'
                    '{"name":"preview","mode":"deep-link","target":"preview://session"}'
                    ']}'
                ),
                encoding='utf-8',
            )
            runtime = RemoteRuntime.from_workspace(workspace)
            report = runtime.connect('staging')
            restored = RemoteRuntime.from_workspace(workspace)

        self.assertEqual(len(runtime.profiles), 2)
        self.assertTrue(report.connected)
        self.assertEqual(report.profile_name, 'staging')
        self.assertIsNotNone(restored.active_connection)
        self.assertEqual(restored.active_connection.profile_name, 'staging')
        self.assertIn('Configured remote profiles: 2', restored.render_summary())

    def test_remote_runtime_disconnect_clears_active_connection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / '.claw-remote.json').write_text(
                '{"profiles":[{"name":"staging","mode":"ssh","target":"dev@staging"}]}',
                encoding='utf-8',
            )
            runtime = RemoteRuntime.from_workspace(workspace)
            runtime.connect('staging')
            report = runtime.disconnect()

        self.assertFalse(report.connected)
        self.assertIn('Disconnected ssh target dev@staging', report.detail)

    def test_remote_mode_helpers_use_manifest_backed_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / '.claw-remote.json').write_text(
                (
                    '{"profiles":['
                    '{"name":"workspace","mode":"remote","target":"remote://workspace"},'
                    '{"name":"sshbox","mode":"ssh","target":"dev@sshbox"},'
                    '{"name":"tele","mode":"teleport","target":"teleport://workspace"},'
                    '{"name":"direct","mode":"direct-connect","target":"direct://workspace"},'
                    '{"name":"link","mode":"deep-link","target":"deep://workspace"}'
                    ']}'
                ),
                encoding='utf-8',
            )
            remote_report = run_remote_mode('workspace', cwd=workspace)
            ssh_report = run_ssh_mode('sshbox', cwd=workspace)
            teleport_report = run_teleport_mode('tele', cwd=workspace)
            direct_report = run_direct_connect_mode('direct', cwd=workspace)
            deep_link_report = run_deep_link_mode('link', cwd=workspace)

        self.assertEqual(remote_report.profile_name, 'workspace')
        self.assertEqual(ssh_report.mode, 'ssh')
        self.assertEqual(teleport_report.mode, 'teleport')
        self.assertEqual(direct_report.mode, 'direct-connect')
        self.assertEqual(deep_link_report.mode, 'deep-link')

    def test_remote_tools_execute_against_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / '.claw-remote.json').write_text(
                (
                    '{"profiles":[{"name":"staging","mode":"ssh","target":"dev@staging",'
                    '"workspaceCwd":"/srv/app","sessionUrl":"wss://remote/session"}]}'
                ),
                encoding='utf-8',
            )
            runtime = RemoteRuntime.from_workspace(workspace)
            context = build_tool_context(
                AgentRuntimeConfig(cwd=workspace),
                remote_runtime=runtime,
            )
            list_result = execute_tool(
                default_tool_registry(),
                'remote_list_profiles',
                {},
                context,
            )
            connect_result = execute_tool(
                default_tool_registry(),
                'remote_connect',
                {'target': 'staging'},
                context,
            )
            status_result = execute_tool(
                default_tool_registry(),
                'remote_status',
                {},
                context,
            )

        self.assertTrue(list_result.ok)
        self.assertIn('staging', list_result.content)
        self.assertTrue(connect_result.ok)
        self.assertIn('profile=staging', connect_result.content)
        self.assertTrue(status_result.ok)
        self.assertIn('Configured remote profiles: 1', status_result.content)
