from __future__ import annotations

import unittest

from src.models import PortingModule
from src.permissions import ToolPermissionContext
from src.tools.tool_pool import ToolPool, assemble_tool_pool


class TestAssembleToolPool(unittest.TestCase):
    def test_returns_tool_pool_with_tools(self) -> None:
        pool = assemble_tool_pool()
        self.assertIsInstance(pool, ToolPool)
        self.assertIsInstance(pool.tools, tuple)
        self.assertTrue(all(isinstance(t, PortingModule) for t in pool.tools))

    def test_default_mode_includes_tools(self) -> None:
        pool = assemble_tool_pool()
        self.assertGreater(len(pool.tools), 0)

    def test_simple_mode_flag_stored(self) -> None:
        pool_default = assemble_tool_pool()
        pool_simple = assemble_tool_pool(simple_mode=True)
        self.assertFalse(pool_default.simple_mode)
        self.assertTrue(pool_simple.simple_mode)

    def test_include_mcp_flag_stored(self) -> None:
        pool_default = assemble_tool_pool()
        pool_no_mcp = assemble_tool_pool(include_mcp=False)
        self.assertTrue(pool_default.include_mcp)
        self.assertFalse(pool_no_mcp.include_mcp)

    def test_simple_mode_reduces_tools(self) -> None:
        pool_full = assemble_tool_pool(simple_mode=False)
        pool_simple = assemble_tool_pool(simple_mode=True)
        self.assertGreater(len(pool_full.tools), len(pool_simple.tools))
        simple_names = {t.name for t in pool_simple.tools}
        self.assertTrue(simple_names.issubset({'BashTool', 'FileReadTool', 'FileEditTool'}))

    def test_include_mcp_false_excludes_mcp_tools(self) -> None:
        pool = assemble_tool_pool(include_mcp=False)
        for tool in pool.tools:
            self.assertNotIn('mcp', tool.name.lower())
            self.assertNotIn('mcp', tool.source_hint.lower())

    def test_permission_context_filters_blocked_tools(self) -> None:
        ctx = ToolPermissionContext.from_iterables(deny_names=['BashTool'])
        pool = assemble_tool_pool(permission_context=ctx)
        tool_names = {t.name for t in pool.tools}
        self.assertNotIn('BashTool', tool_names)

        pool_unfiltered = assemble_tool_pool()
        unfiltered_names = {t.name for t in pool_unfiltered.tools}
        self.assertIn('BashTool', unfiltered_names)


class TestToolPoolAsMarkdown(unittest.TestCase):
    def test_includes_header_and_tool_count(self) -> None:
        pool = assemble_tool_pool()
        md = pool.as_markdown()
        self.assertIn('# Tool Pool', md)
        self.assertIn(f'Tool count: {len(pool.tools)}', md)

    def test_includes_mode_flags(self) -> None:
        pool = assemble_tool_pool(simple_mode=True, include_mcp=False)
        md = pool.as_markdown()
        self.assertIn('Simple mode: True', md)
        self.assertIn('Include MCP: False', md)

    def test_shows_at_most_15_tools(self) -> None:
        pool = assemble_tool_pool()
        self.assertGreater(len(pool.tools), 15, 'Need >15 tools for this test')
        md = pool.as_markdown()
        tool_lines = [line for line in md.splitlines() if line.startswith('- ')]
        self.assertEqual(len(tool_lines), 15)

    def test_empty_tools_renders_correctly(self) -> None:
        pool = ToolPool(tools=(), simple_mode=False, include_mcp=True)
        md = pool.as_markdown()
        self.assertIn('# Tool Pool', md)
        self.assertIn('Tool count: 0', md)
        self.assertIn('Simple mode: False', md)
        self.assertIn('Include MCP: True', md)
        tool_lines = [line for line in md.splitlines() if line.startswith('- ')]
        self.assertEqual(len(tool_lines), 0)

    def test_tool_lines_contain_name_and_source_hint(self) -> None:
        tool = PortingModule(
            name='TestTool',
            responsibility='testing',
            source_hint='test/path.ts',
        )
        pool = ToolPool(tools=(tool,), simple_mode=False, include_mcp=False)
        md = pool.as_markdown()
        self.assertIn('- TestTool — test/path.ts', md)


class TestToolPoolFrozen(unittest.TestCase):
    def test_cannot_mutate_fields(self) -> None:
        pool = ToolPool(tools=(), simple_mode=False, include_mcp=True)
        with self.assertRaises(AttributeError):
            pool.simple_mode = True  # type: ignore[misc]


if __name__ == '__main__':
    unittest.main()
