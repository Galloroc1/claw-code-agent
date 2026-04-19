from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.tools.agent_tools import build_tool_context, default_tool_registry, execute_tool
from src.core.agent_types import AgentRuntimeConfig
from src.search_runtime import SearchRuntime


class FakeHTTPResponse:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload.encode('utf-8')

    def __enter__(self) -> 'FakeHTTPResponse':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class SearchRuntimeTests(unittest.TestCase):
    def test_provider_activation_persists_across_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / '.claw-search.json').write_text(
                (
                    '{"providers":['
                    '{"name":"primary","provider":"searxng","baseUrl":"http://127.0.0.1:8080"},'
                    '{"name":"backup","provider":"searxng","baseUrl":"http://127.0.0.2:8080"}'
                    ']}'
                ),
                encoding='utf-8',
            )
            runtime = SearchRuntime.from_workspace(workspace)
            report = runtime.activate_provider('backup')
            reloaded = SearchRuntime.from_workspace(workspace)

        self.assertEqual(report.provider_name, 'backup')
        self.assertIsNotNone(reloaded.current_provider())
        self.assertEqual(reloaded.current_provider().name, 'backup')

    def test_search_runtime_loads_searxng_provider_from_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            with patch.dict('os.environ', {'SEARXNG_BASE_URL': 'http://127.0.0.1:8888'}, clear=False):
                runtime = SearchRuntime.from_workspace(workspace)

        provider = runtime.current_provider()
        self.assertIsNotNone(provider)
        self.assertEqual(provider.name, 'searxng')
        self.assertEqual(provider.base_url, 'http://127.0.0.1:8888')

    def test_search_runtime_parses_searxng_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / '.claw-search.json').write_text(
                '{"providers":[{"name":"local-search","provider":"searxng","baseUrl":"http://127.0.0.1:8080"}]}',
                encoding='utf-8',
            )
            runtime = SearchRuntime.from_workspace(workspace)
            with patch(
                'src.search_runtime.request.urlopen',
                return_value=FakeHTTPResponse(
                    '{"results":[{"title":"Alpha","url":"https://example.com/alpha","content":"Snippet"}]}'
                ),
            ):
                provider, results = runtime.search('alpha')

        self.assertEqual(provider.name, 'local-search')
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, 'Alpha')
        self.assertEqual(results[0].url, 'https://example.com/alpha')

    def test_search_runtime_parses_brave_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / '.claw-search.json').write_text(
                '{"providers":[{"name":"brave-local","provider":"brave","baseUrl":"https://api.search.brave.com/res/v1/web/search","apiKeyEnv":"BRAVE_SEARCH_API_KEY"}]}',
                encoding='utf-8',
            )
            with patch.dict('os.environ', {'BRAVE_SEARCH_API_KEY': 'demo-key'}, clear=False):
                runtime = SearchRuntime.from_workspace(workspace)
                with patch(
                    'src.search_runtime.request.urlopen',
                    return_value=FakeHTTPResponse(
                        '{"web":{"results":[{"title":"Alpha","url":"https://example.com/alpha","description":"Snippet"}]}}'
                    ),
                ):
                    provider, results = runtime.search('alpha', provider_name='brave-local')

        self.assertEqual(provider.provider, 'brave')
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].snippet, 'Snippet')

    def test_web_search_tool_uses_search_runtime(self) -> None:
        registry = default_tool_registry()
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / '.claw-search.json').write_text(
                '{"providers":[{"name":"local-search","provider":"searxng","baseUrl":"http://127.0.0.1:8080"}]}',
                encoding='utf-8',
            )
            runtime = SearchRuntime.from_workspace(workspace)
            context = build_tool_context(
                AgentRuntimeConfig(cwd=workspace),
                tool_registry=registry,
                search_runtime=runtime,
            )
            with patch(
                'src.search_runtime.request.urlopen',
                return_value=FakeHTTPResponse(
                    '{"results":[{"title":"Alpha","url":"https://example.com/alpha","content":"Snippet"}]}'
                ),
            ):
                result = execute_tool(
                    registry,
                    'web_search',
                    {'query': 'alpha'},
                    context,
                )

        self.assertTrue(result.ok)
        self.assertIn('# Web Search', result.content)
        self.assertEqual(result.metadata.get('action'), 'web_search')
        self.assertEqual(result.metadata.get('provider'), 'local-search')
        self.assertEqual(result.metadata.get('result_count'), 1)
