"""Tests for /files, /copy, /export, /stats, /tag, /rename, /branch, /effort, /doctor."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.core.agent_runtime import LocalCodingAgent
from src.core.agent_session import AgentMessage, AgentSessionState
from src.core.agent_types import AgentRuntimeConfig, ModelConfig, UsageStats


def _make_agent(tmp_dir: str) -> LocalCodingAgent:
    return LocalCodingAgent(
        model_config=ModelConfig(model='test-model'),
        runtime_config=AgentRuntimeConfig(cwd=Path(tmp_dir)),
    )


def _set_session(agent: LocalCodingAgent, messages: list[AgentMessage]) -> None:
    session = AgentSessionState(
        system_prompt_parts=('You are a helper.',),
        messages=messages,
    )
    agent.last_session = session


class TestFilesCommand(unittest.TestCase):
    def test_no_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            result = agent.run('/files')
        self.assertIn('No active session', result.final_output)

    def test_no_files_in_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            _set_session(agent, [
                AgentMessage(role='user', content='Hello'),
                AgentMessage(role='assistant', content='Hi'),
            ])
            result = agent.run('/files')
        self.assertIn('No files loaded', result.final_output)

    def test_files_from_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            _set_session(agent, [
                AgentMessage(role='user', content='Read main.py'),
                AgentMessage(
                    role='assistant', content='',
                    tool_calls=(
                        {'id': 'tc1', 'type': 'function', 'function': {
                            'name': 'Read',
                            'arguments': json.dumps({'file_path': '/home/user/project/main.py'}),
                        }},
                    ),
                ),
                AgentMessage(
                    role='tool', content='print("hello")',
                    name='Read',
                    tool_call_id='tc1',
                    metadata={'path': '/home/user/project/main.py'},
                ),
            ])
            result = agent.run('/files')
        self.assertIn('Files in context', result.final_output)
        self.assertIn('main.py', result.final_output)


class TestCopyCommand(unittest.TestCase):
    def test_no_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            result = agent.run('/copy')
        self.assertIn('No active session', result.final_output)

    def test_no_assistant_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            _set_session(agent, [
                AgentMessage(role='user', content='Hello'),
            ])
            result = agent.run('/copy')
        self.assertIn('No assistant responses', result.final_output)

    def test_copies_latest_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            _set_session(agent, [
                AgentMessage(role='user', content='Hello'),
                AgentMessage(role='assistant', content='First response.'),
                AgentMessage(role='user', content='More'),
                AgentMessage(role='assistant', content='Second response with details.'),
            ])
            result = agent.run('/copy')
        self.assertIn('Copied', result.final_output)
        self.assertIn('response.md', result.final_output)
        # Verify the file was written
        tmp_file = Path(tempfile.gettempdir()) / 'claw-code' / 'response.md'
        self.assertTrue(tmp_file.exists())
        content = tmp_file.read_text()
        self.assertEqual(content, 'Second response with details.')

    def test_copies_nth_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            _set_session(agent, [
                AgentMessage(role='user', content='Hello'),
                AgentMessage(role='assistant', content='First.'),
                AgentMessage(role='user', content='More'),
                AgentMessage(role='assistant', content='Second.'),
            ])
            result = agent.run('/copy 1')
        tmp_file = Path(tempfile.gettempdir()) / 'claw-code' / 'response.md'
        content = tmp_file.read_text()
        self.assertEqual(content, 'First.')


class TestExportCommand(unittest.TestCase):
    def test_no_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            result = agent.run('/export')
        self.assertIn('No active session', result.final_output)

    def test_exports_with_auto_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            _set_session(agent, [
                AgentMessage(role='user', content='Hello'),
                AgentMessage(role='assistant', content='Hi there'),
            ])
            result = agent.run('/export')
        self.assertIn('Exported 2 messages', result.final_output)
        self.assertIn('.txt', result.final_output)

    def test_exports_with_custom_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            _set_session(agent, [
                AgentMessage(role='user', content='Hello'),
                AgentMessage(role='assistant', content='Hi'),
            ])
            result = agent.run('/export my_chat')
            self.assertIn('my_chat.txt', result.final_output)
            out_file = Path(tmp) / 'my_chat.txt'
            self.assertTrue(out_file.exists())
            content = out_file.read_text()
            self.assertIn('Hello', content)
            self.assertIn('Hi', content)


class TestStatsCommand(unittest.TestCase):
    def test_shows_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            _set_session(agent, [
                AgentMessage(role='user', content='Hello'),
                AgentMessage(role='assistant', content='Hi'),
                AgentMessage(role='user', content='Question'),
            ])
            agent.cumulative_usage = UsageStats(input_tokens=500, output_tokens=200)
            result = agent.run('/stats')
        self.assertIn('Session Statistics', result.final_output)
        self.assertIn('3 total', result.final_output)
        self.assertIn('2 user', result.final_output)
        self.assertIn('1 assistant', result.final_output)
        self.assertIn('500', result.final_output)
        self.assertIn('200', result.final_output)


class TestTagCommand(unittest.TestCase):
    def test_no_tags_initially(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            result = agent.run('/tag')
        self.assertIn('No tags set', result.final_output)

    def test_add_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            result = agent.run('/tag important')
        self.assertIn('Added tag: important', result.final_output)

    def test_toggle_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            agent.run('/tag my-tag')
            result = agent.run('/tag my-tag')
        self.assertIn('Removed tag: my-tag', result.final_output)

    def test_list_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            agent.run('/tag alpha')
            agent.run('/tag beta')
            result = agent.run('/tag')
        self.assertIn('alpha', result.final_output)
        self.assertIn('beta', result.final_output)


class TestRenameCommand(unittest.TestCase):
    def test_no_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            result = agent.run('/rename')
        self.assertIn('Usage', result.final_output)

    def test_rename_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            result = agent.run('/rename My Cool Session')
        self.assertIn('renamed to: My Cool Session', result.final_output)


class TestBranchCommand(unittest.TestCase):
    def test_no_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            result = agent.run('/branch')
        self.assertIn('No active session', result.final_output)

    def test_branch_with_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            _set_session(agent, [
                AgentMessage(role='user', content='Hello'),
                AgentMessage(role='assistant', content='Hi'),
            ])
            result = agent.run('/branch my-feature')
        self.assertIn('Created branch "my-feature"', result.final_output)
        self.assertIn('Saved to:', result.final_output)

    def test_branch_auto_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            _set_session(agent, [
                AgentMessage(role='user', content='Hello'),
            ])
            result = agent.run('/branch')
        self.assertIn('Created branch "branch-', result.final_output)


class TestEffortCommand(unittest.TestCase):
    def test_show_current_effort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            result = agent.run('/effort')
        self.assertIn('Current effort level: auto', result.final_output)

    def test_set_effort_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            result = agent.run('/effort high')
        self.assertIn('Set effort level to: high', result.final_output)

    def test_invalid_effort_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            result = agent.run('/effort extreme')
        self.assertIn('Invalid effort level', result.final_output)

    def test_all_valid_levels(self) -> None:
        for level in ('low', 'medium', 'high', 'max', 'auto'):
            with tempfile.TemporaryDirectory() as tmp:
                agent = _make_agent(tmp)
                result = agent.run(f'/effort {level}')
            self.assertIn(f'Set effort level to: {level}', result.final_output)


class TestDoctorCommand(unittest.TestCase):
    def test_shows_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            result = agent.run('/doctor')
        output = result.final_output
        self.assertIn('Doctor Report', output)
        self.assertIn('Python version', output)
        self.assertIn('git', output)
        self.assertIn('Model', output)
        self.assertIn('Working directory', output)

    def test_detects_claude_md(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'CLAUDE.md').write_text('memory file')
            agent = _make_agent(tmp)
            result = agent.run('/doctor')
        self.assertIn('CLAUDE.md', result.final_output)
        self.assertIn('found', result.final_output)


if __name__ == '__main__':
    unittest.main()
