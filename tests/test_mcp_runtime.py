from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.agent_runtime import LocalCodingAgent
from src.tools.agent_tools import build_tool_context, default_tool_registry, execute_tool
from src.core.agent_types import AgentRuntimeConfig, ModelConfig
from src.mcp_runtime import MCPRuntime


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode('utf-8')

    def __enter__(self) -> 'FakeHTTPResponse':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def make_urlopen_side_effect(responses: list[dict[str, object]]):
    queued = [FakeHTTPResponse(payload) for payload in responses]

    def _fake_urlopen(request_obj, timeout=None):  # noqa: ANN001
        return queued.pop(0)

    return _fake_urlopen


class MCPRuntimeTests(unittest.TestCase):
    def _write_fake_stdio_server(self, workspace: Path) -> Path:
        server_path = workspace / 'fake_mcp_server.py'
        server_path.write_text(
            (
                'import json, sys\n'
                'RESOURCES = [{"uri": "mcp://remote/notes", "name": "Remote Notes", "mimeType": "text/plain"}]\n'
                'TOOLS = [{"name": "echo", "description": "Echo text", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}}]\n'
                'for raw in sys.stdin:\n'
                '    raw = raw.strip()\n'
                '    if not raw:\n'
                '        continue\n'
                '    message = json.loads(raw)\n'
                '    method = message.get("method")\n'
                '    if method == "initialize":\n'
                '        response = {"jsonrpc": "2.0", "id": message.get("id"), "result": {"protocolVersion": "2025-11-25", "capabilities": {"resources": {}, "tools": {}}, "serverInfo": {"name": "fake-remote", "version": "1.0.0"}}}\n'
                '        print(json.dumps(response), flush=True)\n'
                '        continue\n'
                '    if method == "notifications/initialized":\n'
                '        continue\n'
                '    if method == "resources/list":\n'
                '        response = {"jsonrpc": "2.0", "id": message.get("id"), "result": {"resources": RESOURCES}}\n'
                '        print(json.dumps(response), flush=True)\n'
                '        continue\n'
                '    if method == "resources/read":\n'
                '        uri = message.get("params", {}).get("uri")\n'
                '        text = "remote notes via stdio" if uri == "mcp://remote/notes" else "unknown resource"\n'
                '        response = {"jsonrpc": "2.0", "id": message.get("id"), "result": {"contents": [{"uri": uri, "mimeType": "text/plain", "text": text}]}}\n'
                '        print(json.dumps(response), flush=True)\n'
                '        continue\n'
                '    if method == "tools/list":\n'
                '        response = {"jsonrpc": "2.0", "id": message.get("id"), "result": {"tools": TOOLS}}\n'
                '        print(json.dumps(response), flush=True)\n'
                '        continue\n'
                '    if method == "tools/call":\n'
                '        params = message.get("params", {})\n'
                '        text = params.get("arguments", {}).get("text", "")\n'
                '        response = {"jsonrpc": "2.0", "id": message.get("id"), "result": {"content": [{"type": "text", "text": "echo:" + text}], "isError": False}}\n'
                '        print(json.dumps(response), flush=True)\n'
                '        continue\n'
                '    response = {"jsonrpc": "2.0", "id": message.get("id"), "error": {"code": -32601, "message": "Method not found"}}\n'
                '    print(json.dumps(response), flush=True)\n'
            ),
            encoding='utf-8',
        )
        return server_path

    def test_runtime_discovers_and_reads_local_resources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / 'notes.txt').write_text('mcp notes\n', encoding='utf-8')
            (workspace / '.claw-mcp.json').write_text(
                (
                    '{"servers":[{"name":"workspace","resources":['
                    '{"uri":"mcp://workspace/notes","name":"Notes","path":"notes.txt"},'
                    '{"uri":"mcp://workspace/inline","name":"Inline","text":"inline body"}'
                    ']}]}'
                ),
                encoding='utf-8',
            )
            runtime = MCPRuntime.from_workspace(workspace)
            self.assertEqual(len(runtime.resources), 2)
            self.assertIn('Local MCP resources: 2', runtime.render_summary())
            self.assertEqual(runtime.read_resource('mcp://workspace/inline'), 'inline body')
            self.assertIn('mcp notes', runtime.read_resource('mcp://workspace/notes'))

    def test_runtime_discovers_stdio_server_and_remote_resources_and_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            server_path = self._write_fake_stdio_server(workspace)
            (workspace / '.claw-mcp.json').write_text(
                json.dumps(
                    {
                        'mcpServers': {
                            'remote': {
                                'command': sys.executable,
                                'args': ['-u', str(server_path)],
                            }
                        }
                    }
                ),
                encoding='utf-8',
            )
            runtime = MCPRuntime.from_workspace(workspace)
            resources = runtime.list_resources()
            tools = runtime.list_tools()
            self.assertEqual(len(runtime.servers), 1)
            self.assertTrue(runtime.has_transport_servers())
            self.assertIn('Configured MCP servers: 1', runtime.render_summary())
            self.assertEqual(len(resources), 1)
            self.assertEqual(resources[0].uri, 'mcp://remote/notes')
            self.assertIn('remote notes via stdio', runtime.read_resource('mcp://remote/notes'))
            self.assertEqual(len(tools), 1)
            self.assertEqual(tools[0].name, 'echo')
            rendered, metadata = runtime.call_tool('echo', arguments={'text': 'hello'})
            self.assertIn('echo:hello', rendered)
            self.assertEqual(metadata.get('server_name'), 'remote')

    def test_mcp_tools_execute_against_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / 'notes.txt').write_text('mcp notes\n', encoding='utf-8')
            (workspace / '.claw-mcp.json').write_text(
                (
                    '{"servers":[{"name":"workspace","resources":['
                    '{"uri":"mcp://workspace/notes","name":"Notes","path":"notes.txt"}'
                    ']}]}'
                ),
                encoding='utf-8',
            )
            runtime = MCPRuntime.from_workspace(workspace)
            context = build_tool_context(
                AgentRuntimeConfig(cwd=workspace),
                mcp_runtime=runtime,
            )
            list_result = execute_tool(
                default_tool_registry(),
                'mcp_list_resources',
                {},
                context,
            )
            read_result = execute_tool(
                default_tool_registry(),
                'mcp_read_resource',
                {'uri': 'mcp://workspace/notes'},
                context,
            )

        self.assertTrue(list_result.ok)
        self.assertIn('mcp://workspace/notes', list_result.content)
        self.assertTrue(read_result.ok)
        self.assertIn('mcp notes', read_result.content)

    def test_mcp_transport_tools_execute_against_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            server_path = self._write_fake_stdio_server(workspace)
            (workspace / '.claw-mcp.json').write_text(
                json.dumps(
                    {
                        'mcpServers': {
                            'remote': {
                                'command': sys.executable,
                                'args': ['-u', str(server_path)],
                            }
                        }
                    }
                ),
                encoding='utf-8',
            )
            runtime = MCPRuntime.from_workspace(workspace)
            context = build_tool_context(
                AgentRuntimeConfig(cwd=workspace),
                mcp_runtime=runtime,
            )
            list_result = execute_tool(
                default_tool_registry(),
                'mcp_list_tools',
                {},
                context,
            )
            call_result = execute_tool(
                default_tool_registry(),
                'mcp_call_tool',
                {'tool_name': 'echo', 'arguments': {'text': 'tool-run'}},
                context,
            )

        self.assertTrue(list_result.ok)
        self.assertIn('echo', list_result.content)
        self.assertTrue(call_result.ok)
        self.assertIn('echo:tool-run', call_result.content)
        self.assertEqual(call_result.metadata.get('action'), 'mcp_call_tool')

    def test_agent_can_use_mcp_tools_in_model_loop(self) -> None:
        responses = [
            {
                'choices': [
                    {
                        'message': {
                            'role': 'assistant',
                            'content': 'I will inspect the MCP resource.',
                            'tool_calls': [
                                {
                                    'id': 'call_1',
                                    'type': 'function',
                                    'function': {
                                        'name': 'mcp_read_resource',
                                        'arguments': '{"uri": "mcp://workspace/notes"}',
                                    },
                                }
                            ],
                        },
                        'finish_reason': 'tool_calls',
                    }
                ],
                'usage': {'prompt_tokens': 8, 'completion_tokens': 3},
            },
            {
                'choices': [
                    {
                        'message': {
                            'role': 'assistant',
                            'content': 'The MCP resource says mcp notes.',
                        },
                        'finish_reason': 'stop',
                    }
                ],
                'usage': {'prompt_tokens': 6, 'completion_tokens': 3},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / 'notes.txt').write_text('mcp notes\n', encoding='utf-8')
            (workspace / '.claw-mcp.json').write_text(
                (
                    '{"servers":[{"name":"workspace","resources":['
                    '{"uri":"mcp://workspace/notes","name":"Notes","path":"notes.txt"}'
                    ']}]}'
                ),
                encoding='utf-8',
            )
            with patch('src.openai_compat.request.urlopen', side_effect=make_urlopen_side_effect(responses)):
                agent = LocalCodingAgent(
                    model_config=ModelConfig(
                        model='Qwen/Qwen3-Coder-30B-A3B-Instruct',
                        base_url='http://127.0.0.1:8000/v1',
                    ),
                    runtime_config=AgentRuntimeConfig(cwd=workspace),
                )
                result = agent.run('Read the MCP notes resource')

        self.assertEqual(result.final_output, 'The MCP resource says mcp notes.')
        self.assertEqual(result.tool_calls, 1)
        tool_message = next(
            message
            for message in result.transcript
            if message.get('role') == 'tool'
        )
        self.assertIn('mcp notes', tool_message.get('content', ''))

    def test_agent_can_use_transport_backed_mcp_call_tool_in_model_loop(self) -> None:
        responses = [
            {
                'choices': [
                    {
                        'message': {
                            'role': 'assistant',
                            'content': 'I will call the remote MCP tool.',
                            'tool_calls': [
                                {
                                    'id': 'call_1',
                                    'type': 'function',
                                    'function': {
                                        'name': 'mcp_call_tool',
                                        'arguments': '{"tool_name": "echo", "server": "remote", "arguments": {"text": "agent-call"}}',
                                    },
                                }
                            ],
                        },
                        'finish_reason': 'tool_calls',
                    }
                ],
                'usage': {'prompt_tokens': 8, 'completion_tokens': 3},
            },
            {
                'choices': [
                    {
                        'message': {
                            'role': 'assistant',
                            'content': 'The remote MCP tool replied with echo:agent-call.',
                        },
                        'finish_reason': 'stop',
                    }
                ],
                'usage': {'prompt_tokens': 6, 'completion_tokens': 3},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            server_path = self._write_fake_stdio_server(workspace)
            (workspace / '.claw-mcp.json').write_text(
                json.dumps(
                    {
                        'mcpServers': {
                            'remote': {
                                'command': sys.executable,
                                'args': ['-u', str(server_path)],
                            }
                        }
                    }
                ),
                encoding='utf-8',
            )
            with patch('src.openai_compat.request.urlopen', side_effect=make_urlopen_side_effect(responses)):
                agent = LocalCodingAgent(
                    model_config=ModelConfig(
                        model='Qwen/Qwen3-Coder-30B-A3B-Instruct',
                        base_url='http://127.0.0.1:8000/v1',
                    ),
                    runtime_config=AgentRuntimeConfig(cwd=workspace),
                )
                result = agent.run('Call the remote MCP echo tool')

        self.assertEqual(result.final_output, 'The remote MCP tool replied with echo:agent-call.')
        self.assertEqual(result.tool_calls, 1)
        tool_message = next(
            message
            for message in result.transcript
            if message.get('role') == 'tool'
        )
        self.assertIn('echo:agent-call', tool_message.get('content', ''))
