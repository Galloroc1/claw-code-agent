"""Tests for /cost, /exit, and /diff slash commands."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from src.core.agent_runtime import LocalCodingAgent
from src.core.agent_types import AgentRuntimeConfig, ModelConfig, UsageStats


class TestCostCommand(unittest.TestCase):
    """Tests for the /cost slash command."""

    def test_cost_shows_zero_initially(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='test-model'),
                runtime_config=AgentRuntimeConfig(cwd=Path(tmp_dir)),
            )
            result = agent.run('/cost')
        self.assertIn('Total cost:', result.final_output)
        self.assertIn('$0.0000', result.final_output)
        self.assertIn('Total input tokens:', result.final_output)
        self.assertIn('Total output tokens:', result.final_output)
        self.assertIn('Total tokens:', result.final_output)

    def test_cost_shows_accumulated_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='test-model'),
                runtime_config=AgentRuntimeConfig(cwd=Path(tmp_dir)),
            )
            # Simulate accumulated usage
            agent.cumulative_usage = UsageStats(
                input_tokens=1000,
                output_tokens=500,
                cache_read_input_tokens=200,
            )
            agent.cumulative_cost_usd = 0.05
            result = agent.run('/cost')
        self.assertIn('$0.05', result.final_output)
        self.assertIn('1,000', result.final_output)
        self.assertIn('500', result.final_output)
        self.assertIn('Cache read tokens:', result.final_output)

    def test_cost_hides_zero_cache_and_reasoning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='test-model'),
                runtime_config=AgentRuntimeConfig(cwd=Path(tmp_dir)),
            )
            agent.cumulative_usage = UsageStats(
                input_tokens=100,
                output_tokens=50,
            )
            result = agent.run('/cost')
        # Should NOT show cache/reasoning lines when they're zero
        self.assertNotIn('Cache read tokens:', result.final_output)
        self.assertNotIn('Cache creation tokens:', result.final_output)
        self.assertNotIn('Reasoning tokens:', result.final_output)

    def test_cost_small_amounts_show_four_decimals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='test-model'),
                runtime_config=AgentRuntimeConfig(cwd=Path(tmp_dir)),
            )
            agent.cumulative_cost_usd = 0.0023
            result = agent.run('/cost')
        self.assertIn('$0.0023', result.final_output)


class TestExitCommand(unittest.TestCase):
    """Tests for the /exit and /quit slash commands."""

    def test_exit_triggers_system_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='test-model'),
                runtime_config=AgentRuntimeConfig(cwd=Path(tmp_dir)),
            )
            with self.assertRaises(SystemExit) as cm:
                agent.run('/exit')
            self.assertEqual(cm.exception.code, 0)

    def test_quit_triggers_system_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='test-model'),
                runtime_config=AgentRuntimeConfig(cwd=Path(tmp_dir)),
            )
            with self.assertRaises(SystemExit):
                agent.run('/quit')


class TestDiffCommand(unittest.TestCase):
    """Tests for the /diff slash command."""

    def test_diff_in_non_git_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='test-model'),
                runtime_config=AgentRuntimeConfig(cwd=Path(tmp_dir)),
            )
            result = agent.run('/diff')
        # In a non-git dir, git diff returns an error
        output = result.final_output.lower()
        self.assertTrue(
            'no uncommitted' in output
            or 'not a git' in output
            or 'error' in output,
            f'Expected a non-git or no-changes message, got: {result.final_output}',
        )

    def test_diff_in_git_repo_with_no_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            subprocess.run(
                ['git', 'init'], cwd=str(workspace),
                capture_output=True, check=True,
            )
            subprocess.run(
                ['git', 'config', 'user.email', 'test@test.com'],
                cwd=str(workspace), capture_output=True,
            )
            subprocess.run(
                ['git', 'config', 'user.name', 'Test'],
                cwd=str(workspace), capture_output=True,
            )
            (workspace / 'hello.txt').write_text('hello\n')
            subprocess.run(
                ['git', 'add', '.'], cwd=str(workspace),
                capture_output=True, check=True,
            )
            subprocess.run(
                ['git', 'commit', '-m', 'init'], cwd=str(workspace),
                capture_output=True, check=True,
            )

            agent = LocalCodingAgent(
                model_config=ModelConfig(model='test-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace),
            )
            result = agent.run('/diff')
        self.assertIn('No uncommitted changes', result.final_output)

    def test_diff_shows_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            subprocess.run(
                ['git', 'init'], cwd=str(workspace),
                capture_output=True, check=True,
            )
            subprocess.run(
                ['git', 'config', 'user.email', 'test@test.com'],
                cwd=str(workspace), capture_output=True,
            )
            subprocess.run(
                ['git', 'config', 'user.name', 'Test'],
                cwd=str(workspace), capture_output=True,
            )
            (workspace / 'hello.txt').write_text('hello\n')
            subprocess.run(
                ['git', 'add', '.'], cwd=str(workspace),
                capture_output=True, check=True,
            )
            subprocess.run(
                ['git', 'commit', '-m', 'init'], cwd=str(workspace),
                capture_output=True, check=True,
            )
            # Make a change
            (workspace / 'hello.txt').write_text('hello world\n')

            agent = LocalCodingAgent(
                model_config=ModelConfig(model='test-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace),
            )
            result = agent.run('/diff')
        self.assertIn('hello', result.final_output)
        self.assertIn('diff', result.final_output.lower())


if __name__ == '__main__':
    unittest.main()
