from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.tools.agent_tools import build_tool_context, default_tool_registry, execute_tool
from src.core.agent_types import AgentPermissions, AgentRuntimeConfig
from src.lsp_runtime import LSPRuntime


class ExtendedToolTests(unittest.TestCase):
    def test_web_fetch_rejects_file_url(self) -> None:
        """Verify that file:// URLs are blocked to prevent SSRF attacks."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            target = workspace / 'page.txt'
            target.write_text('hello from web fetch\n', encoding='utf-8')
            context = build_tool_context(
                AgentRuntimeConfig(cwd=workspace),
                tool_registry=default_tool_registry(),
            )
            result = execute_tool(
                default_tool_registry(),
                'web_fetch',
                {'url': target.resolve().as_uri()},
                context,
            )

        self.assertFalse(result.ok)
        self.assertIn('http or https', result.content)

    def test_tool_search_lists_matching_tools(self) -> None:
        registry = default_tool_registry()
        with tempfile.TemporaryDirectory() as tmp_dir:
            context = build_tool_context(
                AgentRuntimeConfig(cwd=Path(tmp_dir)),
                tool_registry=registry,
            )
            result = execute_tool(
                registry,
                'tool_search',
                {'query': 'file'},
                context,
            )

        self.assertTrue(result.ok)
        self.assertIn('# Tool Search', result.content)
        self.assertIn('read_file', result.content)
        self.assertIn('write_file', result.content)

    def test_sleep_tool_waits_briefly_and_returns_metadata(self) -> None:
        registry = default_tool_registry()
        with tempfile.TemporaryDirectory() as tmp_dir:
            context = build_tool_context(
                AgentRuntimeConfig(cwd=Path(tmp_dir)),
                tool_registry=registry,
            )
            result = execute_tool(
                registry,
                'sleep',
                {'seconds': 0.01},
                context,
            )

        self.assertTrue(result.ok)
        self.assertIn('slept for', result.content)
        self.assertEqual(result.metadata.get('action'), 'sleep')

    def test_notebook_edit_updates_ipynb_cell(self) -> None:
        registry = default_tool_registry()
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            notebook = workspace / 'demo.ipynb'
            notebook.write_text(
                '{\n'
                ' "cells": [\n'
                '  {"cell_type": "code", "metadata": {}, "source": ["print(1)\\n"], "outputs": [], "execution_count": null}\n'
                ' ],\n'
                ' "metadata": {},\n'
                ' "nbformat": 4,\n'
                ' "nbformat_minor": 5\n'
                '}\n',
                encoding='utf-8',
            )
            context = build_tool_context(
                AgentRuntimeConfig(
                    cwd=workspace,
                    permissions=AgentPermissions(allow_file_write=True),
                ),
                tool_registry=registry,
            )
            result = execute_tool(
                registry,
                'notebook_edit',
                {'path': 'demo.ipynb', 'cell_index': 0, 'source': 'print(2)\n'},
                context,
            )
            updated = notebook.read_text(encoding='utf-8')

        self.assertTrue(result.ok)
        self.assertIn('updated notebook cell 0', result.content)
        self.assertIn('print(2)', updated)
        self.assertEqual(result.metadata.get('action'), 'notebook_edit')

    def test_lsp_tool_returns_definition_report(self) -> None:
        registry = default_tool_registry()
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / 'sample.py').write_text(
                'def helper(value):\n'
                '    return value * 2\n'
                '\n'
                'def run(item):\n'
                '    return helper(item)\n',
                encoding='utf-8',
            )
            context = build_tool_context(
                AgentRuntimeConfig(cwd=workspace),
                tool_registry=registry,
                lsp_runtime=LSPRuntime.from_workspace(workspace),
            )
            result = execute_tool(
                registry,
                'LSP',
                {
                    'operation': 'goToDefinition',
                    'file_path': 'sample.py',
                    'line': 5,
                    'character': 12,
                },
                context,
            )

        self.assertTrue(result.ok)
        self.assertIn('# LSP Definition', result.content)
        self.assertIn('function helper', result.content)
        self.assertEqual(result.metadata.get('action'), 'lsp_query')
        self.assertEqual(result.metadata.get('operation'), 'goToDefinition')
