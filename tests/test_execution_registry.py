from __future__ import annotations

import unittest

from src.execution_registry import (
    ExecutionRegistry,
    MirroredCommand,
    MirroredTool,
    build_execution_registry,
)
from src.commands import PORTED_COMMANDS
from src.tools import PORTED_TOOLS


class TestBuildExecutionRegistry(unittest.TestCase):
    """Tests for build_execution_registry and basic registry properties."""

    def test_returns_execution_registry(self) -> None:
        registry = build_execution_registry()
        self.assertIsInstance(registry, ExecutionRegistry)

    def test_has_non_empty_commands(self) -> None:
        registry = build_execution_registry()
        self.assertGreater(len(registry.commands), 0)

    def test_has_non_empty_tools(self) -> None:
        registry = build_execution_registry()
        self.assertGreater(len(registry.tools), 0)

    def test_command_count_matches_ported_commands(self) -> None:
        registry = build_execution_registry()
        self.assertEqual(len(registry.commands), len(PORTED_COMMANDS))

    def test_tool_count_matches_ported_tools(self) -> None:
        registry = build_execution_registry()
        self.assertEqual(len(registry.tools), len(PORTED_TOOLS))


class TestCommandLookup(unittest.TestCase):
    """Tests for ExecutionRegistry.command() lookup."""

    def setUp(self) -> None:
        self.registry = build_execution_registry()
        self.known_name = self.registry.commands[0].name

    def test_lookup_exact_case(self) -> None:
        result = self.registry.command(self.known_name)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, self.known_name)

    def test_lookup_case_insensitive_lower(self) -> None:
        result = self.registry.command(self.known_name.lower())
        self.assertIsNotNone(result)
        self.assertEqual(result.name, self.known_name)

    def test_lookup_case_insensitive_upper(self) -> None:
        result = self.registry.command(self.known_name.upper())
        self.assertIsNotNone(result)
        self.assertEqual(result.name, self.known_name)

    def test_lookup_case_insensitive_mixed(self) -> None:
        mixed = ''.join(
            c.upper() if i % 2 else c.lower()
            for i, c in enumerate(self.known_name)
        )
        result = self.registry.command(mixed)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, self.known_name)

    def test_returns_none_for_unknown_name(self) -> None:
        result = self.registry.command('__nonexistent_command_xyz__')
        self.assertIsNone(result)


class TestToolLookup(unittest.TestCase):
    """Tests for ExecutionRegistry.tool() lookup."""

    def setUp(self) -> None:
        self.registry = build_execution_registry()
        self.known_name = self.registry.tools[0].name

    def test_lookup_exact_case(self) -> None:
        result = self.registry.tool(self.known_name)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, self.known_name)

    def test_lookup_case_insensitive_lower(self) -> None:
        result = self.registry.tool(self.known_name.lower())
        self.assertIsNotNone(result)
        self.assertEqual(result.name, self.known_name)

    def test_lookup_case_insensitive_upper(self) -> None:
        result = self.registry.tool(self.known_name.upper())
        self.assertIsNotNone(result)
        self.assertEqual(result.name, self.known_name)

    def test_lookup_case_insensitive_mixed(self) -> None:
        mixed = ''.join(
            c.upper() if i % 2 else c.lower()
            for i, c in enumerate(self.known_name)
        )
        result = self.registry.tool(mixed)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, self.known_name)

    def test_returns_none_for_unknown_name(self) -> None:
        result = self.registry.tool('__nonexistent_tool_xyz__')
        self.assertIsNone(result)


class TestMirroredCommand(unittest.TestCase):
    """Tests for MirroredCommand dataclass."""

    def test_has_correct_name(self) -> None:
        cmd = MirroredCommand(name='review', source_hint='copilot')
        self.assertEqual(cmd.name, 'review')

    def test_has_correct_source_hint(self) -> None:
        cmd = MirroredCommand(name='review', source_hint='copilot')
        self.assertEqual(cmd.source_hint, 'copilot')

    def test_is_frozen(self) -> None:
        cmd = MirroredCommand(name='review', source_hint='copilot')
        with self.assertRaises(AttributeError):
            cmd.name = 'other'  # type: ignore[misc]

    def test_execute_returns_string(self) -> None:
        registry = build_execution_registry()
        cmd = registry.commands[0]
        result = cmd.execute('test prompt')
        self.assertIsInstance(result, str)
        self.assertIn('Mirrored command', result)


class TestMirroredTool(unittest.TestCase):
    """Tests for MirroredTool dataclass."""

    def test_has_correct_name(self) -> None:
        tool = MirroredTool(name='BashTool', source_hint='vscode')
        self.assertEqual(tool.name, 'BashTool')

    def test_has_correct_source_hint(self) -> None:
        tool = MirroredTool(name='BashTool', source_hint='vscode')
        self.assertEqual(tool.source_hint, 'vscode')

    def test_is_frozen(self) -> None:
        tool = MirroredTool(name='BashTool', source_hint='vscode')
        with self.assertRaises(AttributeError):
            tool.name = 'other'  # type: ignore[misc]

    def test_execute_returns_string(self) -> None:
        registry = build_execution_registry()
        tool = registry.tools[0]
        result = tool.execute('test payload')
        self.assertIsInstance(result, str)
        self.assertIn('Mirrored tool', result)


class TestEmptyRegistry(unittest.TestCase):
    """Tests for an empty ExecutionRegistry."""

    def setUp(self) -> None:
        self.registry = ExecutionRegistry(commands=(), tools=())

    def test_command_returns_none(self) -> None:
        self.assertIsNone(self.registry.command('anything'))

    def test_tool_returns_none(self) -> None:
        self.assertIsNone(self.registry.tool('anything'))

    def test_empty_commands_tuple(self) -> None:
        self.assertEqual(len(self.registry.commands), 0)

    def test_empty_tools_tuple(self) -> None:
        self.assertEqual(len(self.registry.tools), 0)


if __name__ == '__main__':
    unittest.main()
