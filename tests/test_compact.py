"""Tests for src/compact.py – the conversation compaction service."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.core.agent_runtime import LocalCodingAgent
from src.core.agent_session import AgentMessage, AgentSessionState
from src.core.agent_types import AgentRuntimeConfig, ModelConfig
from src.compact import (
    AUTOCOMPACT_BUFFER_TOKENS,
    compact_conversation,
    format_compact_summary,
    get_compact_prompt,
    get_compact_user_summary_message,
)


class TestGetCompactPrompt(unittest.TestCase):
    """Tests for the compact prompt builder."""

    def test_basic_prompt_contains_all_nine_sections(self) -> None:
        prompt = get_compact_prompt()
        for section in [
            '1. Primary Request and Intent',
            '2. Key Technical Concepts',
            '3. Files and Code Sections',
            '4. Errors and fixes',
            '5. Problem Solving',
            '6. All user messages',
            '7. Pending Tasks',
            '8. Current Work',
            '9. Optional Next Step',
        ]:
            self.assertIn(section, prompt, f'Missing section: {section}')

    def test_no_tools_preamble_present(self) -> None:
        prompt = get_compact_prompt()
        self.assertIn('CRITICAL: Respond with TEXT ONLY', prompt)
        self.assertIn('Do NOT call any tools', prompt)

    def test_no_tools_trailer_present(self) -> None:
        prompt = get_compact_prompt()
        self.assertIn('REMINDER: Do NOT call any tools', prompt)

    def test_analysis_and_summary_example_tags_present(self) -> None:
        prompt = get_compact_prompt()
        self.assertIn('<analysis>', prompt)
        self.assertIn('</analysis>', prompt)
        self.assertIn('<summary>', prompt)
        self.assertIn('</summary>', prompt)

    def test_custom_instructions_appended(self) -> None:
        prompt = get_compact_prompt('Focus on database changes.')
        self.assertIn('Additional Instructions:', prompt)
        self.assertIn('Focus on database changes.', prompt)

    def test_empty_custom_instructions_ignored(self) -> None:
        prompt_no_custom = get_compact_prompt()
        prompt_empty = get_compact_prompt('')
        prompt_whitespace = get_compact_prompt('   ')
        self.assertEqual(prompt_no_custom, prompt_empty)
        self.assertEqual(prompt_no_custom, prompt_whitespace)

    def test_none_custom_instructions_ignored(self) -> None:
        prompt_none = get_compact_prompt(None)
        prompt_no_arg = get_compact_prompt()
        self.assertEqual(prompt_none, prompt_no_arg)


class TestFormatCompactSummary(unittest.TestCase):
    """Tests for the summary formatting / XML stripping."""

    def test_strips_analysis_block(self) -> None:
        raw = '<analysis>thinking here</analysis>\n\n<summary>result</summary>'
        formatted = format_compact_summary(raw)
        self.assertNotIn('<analysis>', formatted)
        self.assertNotIn('thinking here', formatted)
        self.assertIn('result', formatted)

    def test_unwraps_summary_tags(self) -> None:
        raw = '<summary>The main points.\n1. First</summary>'
        formatted = format_compact_summary(raw)
        self.assertNotIn('<summary>', formatted)
        self.assertNotIn('</summary>', formatted)
        self.assertIn('Summary:', formatted)
        self.assertIn('The main points.', formatted)

    def test_handles_no_xml_tags(self) -> None:
        raw = 'Plain text summary without any tags.'
        formatted = format_compact_summary(raw)
        self.assertEqual(formatted, raw)

    def test_collapses_excess_blank_lines(self) -> None:
        raw = '<analysis>x</analysis>\n\n\n\n<summary>y</summary>'
        formatted = format_compact_summary(raw)
        self.assertNotIn('\n\n\n', formatted)

    def test_multiline_analysis_stripped(self) -> None:
        raw = (
            '<analysis>\nLine 1\nLine 2\nLine 3\n</analysis>\n'
            '<summary>Final summary</summary>'
        )
        formatted = format_compact_summary(raw)
        self.assertNotIn('Line 1', formatted)
        self.assertIn('Final summary', formatted)


class TestGetCompactUserSummaryMessage(unittest.TestCase):
    """Tests for the post-compact user message builder."""

    def test_basic_message_structure(self) -> None:
        msg = get_compact_user_summary_message('<summary>overview</summary>')
        self.assertIn('continued from a previous conversation', msg)
        self.assertIn('overview', msg)

    def test_transcript_path_appended(self) -> None:
        msg = get_compact_user_summary_message(
            '<summary>ok</summary>',
            transcript_path='/tmp/transcript.json',
        )
        self.assertIn('/tmp/transcript.json', msg)

    def test_suppress_follow_up(self) -> None:
        msg = get_compact_user_summary_message(
            '<summary>ok</summary>',
            suppress_follow_up=True,
        )
        self.assertIn('without asking the user any further questions', msg)

    def test_no_suppress_follow_up_default(self) -> None:
        msg = get_compact_user_summary_message('<summary>ok</summary>')
        self.assertNotIn('without asking the user any further questions', msg)


class TestCompactConversation(unittest.TestCase):
    """Tests for the core compact_conversation() function."""

    def _make_agent(self, tmp_dir: str) -> LocalCodingAgent:
        """Create a minimal agent with a session loaded."""
        agent = LocalCodingAgent(
            model_config=ModelConfig(model='test-model'),
            runtime_config=AgentRuntimeConfig(cwd=Path(tmp_dir)),
        )
        return agent

    def _set_session(
        self, agent: LocalCodingAgent, messages: list[AgentMessage]
    ) -> None:
        session = AgentSessionState(
            system_prompt_parts=('You are a helpful assistant.',),
            messages=messages,
        )
        agent.last_session = session

    def _make_messages(self, count: int) -> list[AgentMessage]:
        msgs: list[AgentMessage] = []
        for i in range(count):
            role = 'user' if i % 2 == 0 else 'assistant'
            msgs.append(
                AgentMessage(
                    role=role,
                    content=f'Message {i} content. ' * 10,
                    message_id=f'msg_{i}',
                )
            )
        return msgs

    def test_no_session_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = self._make_agent(tmp_dir)
            agent.last_session = None
            result = compact_conversation(agent)
        self.assertIsNotNone(result.error)
        self.assertIn('Not enough', result.error)

    def test_empty_session_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = self._make_agent(tmp_dir)
            self._set_session(agent, [])
            result = compact_conversation(agent)
        self.assertIsNotNone(result.error)

    def test_too_few_messages_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = self._make_agent(tmp_dir)
            # With only 2 messages and preserve_count=4, nothing to compact
            self._set_session(agent, self._make_messages(2))
            result = compact_conversation(agent)
        self.assertIsNotNone(result.error)

    def test_successful_compaction(self) -> None:
        """Simulate a successful model call and verify session is compacted."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = self._make_agent(tmp_dir)
            msgs = self._make_messages(10)
            self._set_session(agent, msgs)

            # Mock the client's complete method
            from src.openai_compat import AssistantTurn
            from src.core.agent_types import UsageStats

            mock_turn = AssistantTurn(
                content=(
                    '<analysis>Thinking through the conversation...</analysis>\n'
                    '<summary>\n1. Primary Request and Intent:\n'
                    '   User wanted to test compaction.\n'
                    '2. Key Technical Concepts:\n   - Testing\n'
                    '3. Files and Code Sections:\n   - test.py\n'
                    '4. Errors and fixes:\n   - None\n'
                    '5. Problem Solving:\n   Basic testing.\n'
                    '6. All user messages:\n   - "test compaction"\n'
                    '7. Pending Tasks:\n   - None\n'
                    '8. Current Work:\n   Testing compaction.\n'
                    '9. Optional Next Step:\n   Verify it works.\n'
                    '</summary>'
                ),
                tool_calls=(),
                finish_reason='stop',
                raw_message={},
                usage=UsageStats(),
            )
            agent.client = MagicMock()
            agent.client.complete.return_value = mock_turn

            result = compact_conversation(agent)

        self.assertIsNone(result.error)
        self.assertGreater(result.pre_compact_token_count, 0)
        # Session should have fewer messages than original 10
        self.assertLess(
            len(agent.last_session.messages), 10,
            'Session should have fewer messages after compaction',
        )
        # Should contain a compact_boundary message
        boundary_msgs = [
            m for m in agent.last_session.messages
            if m.metadata.get('kind') == 'compact_boundary'
        ]
        self.assertEqual(len(boundary_msgs), 1)
        # Should contain a compact_summary message
        summary_msgs = [
            m for m in agent.last_session.messages
            if m.metadata.get('kind') == 'compact_summary'
        ]
        self.assertEqual(len(summary_msgs), 1)
        # Summary should not contain <analysis> block
        self.assertNotIn('<analysis>', result.summary_text)
        # Summary should contain the actual summary content
        self.assertIn('User wanted to test compaction', result.summary_text)

    def test_api_error_returns_compaction_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = self._make_agent(tmp_dir)
            self._set_session(agent, self._make_messages(10))

            agent.client = MagicMock()
            agent.client.complete.side_effect = RuntimeError('API down')

            result = compact_conversation(agent)
        self.assertIsNotNone(result.error)
        self.assertIn('API down', result.error)

    def test_empty_model_response_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = self._make_agent(tmp_dir)
            self._set_session(agent, self._make_messages(10))

            from src.openai_compat import AssistantTurn
            from src.core.agent_types import UsageStats

            agent.client = MagicMock()
            agent.client.complete.return_value = AssistantTurn(
                content='',
                tool_calls=(),
                finish_reason='stop',
                raw_message={},
                usage=UsageStats(),
            )

            result = compact_conversation(agent)
        self.assertIsNotNone(result.error)

    def test_preserves_tail_messages(self) -> None:
        """The most recent messages should survive compaction."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = self._make_agent(tmp_dir)
            msgs = self._make_messages(12)
            self._set_session(agent, msgs)

            from src.openai_compat import AssistantTurn
            from src.core.agent_types import UsageStats

            agent.client = MagicMock()
            agent.client.complete.return_value = AssistantTurn(
                content='<summary>Summarised.</summary>',
                tool_calls=(),
                finish_reason='stop',
                raw_message={},
                usage=UsageStats(),
            )

            result = compact_conversation(agent)

        self.assertIsNone(result.error)
        # The last 4 messages (default preserve_count) should still be present
        session_contents = [m.content for m in agent.last_session.messages]
        for original_msg in msgs[-4:]:
            self.assertIn(
                original_msg.content,
                session_contents,
                f'Tail message "{original_msg.message_id}" should be preserved',
            )

    def test_custom_instructions_passed_to_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = self._make_agent(tmp_dir)
            self._set_session(agent, self._make_messages(10))

            from src.openai_compat import AssistantTurn
            from src.core.agent_types import UsageStats

            agent.client = MagicMock()
            agent.client.complete.return_value = AssistantTurn(
                content='<summary>Custom summary.</summary>',
                tool_calls=(),
                finish_reason='stop',
                raw_message={},
                usage=UsageStats(),
            )

            compact_conversation(agent, custom_instructions='Focus on CSS.')

        # Check that the API was called with custom instructions in the prompt
        call_args = agent.client.complete.call_args
        messages = call_args[0][0]
        last_user_msg = messages[-1]['content']
        self.assertIn('Focus on CSS.', last_user_msg)
        self.assertIn('Additional Instructions:', last_user_msg)


class TestCompactSlashCommand(unittest.TestCase):
    """Test the /compact slash command handler end-to-end."""

    def test_slash_compact_returns_success_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='test-model'),
                runtime_config=AgentRuntimeConfig(cwd=Path(tmp_dir)),
            )
            # Set up a session with enough messages
            session = AgentSessionState(
                system_prompt_parts=('You are a helper.',),
                messages=[
                    AgentMessage(role='user', content='Hello', message_id=f'u{i}')
                    if i % 2 == 0
                    else AgentMessage(role='assistant', content='Hi', message_id=f'a{i}')
                    for i in range(10)
                ],
            )
            agent.last_session = session

            from src.openai_compat import AssistantTurn
            from src.core.agent_types import UsageStats

            agent.client = MagicMock()
            agent.client.complete.return_value = AssistantTurn(
                content='<summary>Session summarised.</summary>',
                tool_calls=(),
                finish_reason='stop',
                raw_message={},
                usage=UsageStats(),
            )

            result = agent.run('/compact')

        self.assertIn('Conversation compacted', result.final_output)
        self.assertIn('Tokens before', result.final_output)

    def test_slash_compact_no_session_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='test-model'),
                runtime_config=AgentRuntimeConfig(cwd=Path(tmp_dir)),
            )
            result = agent.run('/compact')
        self.assertIn('failed', result.final_output.lower())


class TestConstants(unittest.TestCase):
    """Verify key constants match the npm reference."""

    def test_autocompact_buffer_tokens(self) -> None:
        self.assertEqual(AUTOCOMPACT_BUFFER_TOKENS, 13_000)


if __name__ == '__main__':
    unittest.main()
