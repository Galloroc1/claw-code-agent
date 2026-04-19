from __future__ import annotations

import unittest

from src.openai_compat import (
    OpenAICompatError,
    _build_response_format,
    _join_url,
    _normalize_content,
    _optional_int,
    _parse_tool_arguments,
    _parse_usage,
)
from src.core.agent_types import OutputSchemaConfig, UsageStats


class TestJoinUrl(unittest.TestCase):
    def test_base_with_trailing_slash(self):
        self.assertEqual(_join_url('http://localhost:8000/', 'v1/chat'), 'http://localhost:8000/v1/chat')

    def test_base_without_trailing_slash(self):
        self.assertEqual(_join_url('http://localhost:8000', 'v1/chat'), 'http://localhost:8000/v1/chat')

    def test_suffix_with_leading_slash(self):
        self.assertEqual(_join_url('http://localhost:8000', '/v1/chat'), 'http://localhost:8000/v1/chat')


class TestNormalizeContent(unittest.TestCase):
    def test_string_passthrough(self):
        self.assertEqual(_normalize_content('hello'), 'hello')

    def test_none_returns_empty(self):
        self.assertEqual(_normalize_content(None), '')

    def test_list_of_strings_joined(self):
        self.assertEqual(_normalize_content(['hello', ' ', 'world']), 'hello world')

    def test_list_of_text_dicts(self):
        items = [{'type': 'text', 'text': 'hello'}, {'type': 'text', 'text': ' world'}]
        self.assertEqual(_normalize_content(items), 'hello world')

    def test_list_of_mixed_items(self):
        items = ['start ', {'type': 'text', 'text': 'middle'}, ' end']
        self.assertEqual(_normalize_content(items), 'start middle end')

    def test_non_string_non_list_returns_str(self):
        self.assertEqual(_normalize_content(42), '42')


class TestParseToolArguments(unittest.TestCase):
    def test_dict_passthrough(self):
        d = {'key': 'value'}
        self.assertIs(_parse_tool_arguments(d), d)

    def test_valid_json_string(self):
        self.assertEqual(_parse_tool_arguments('{"a": 1}'), {'a': 1})

    def test_empty_string_returns_empty_dict(self):
        self.assertEqual(_parse_tool_arguments(''), {})

    def test_none_returns_empty_dict(self):
        self.assertEqual(_parse_tool_arguments(None), {})

    def test_invalid_json_raises(self):
        with self.assertRaises(OpenAICompatError):
            _parse_tool_arguments('{bad json}')

    def test_json_non_dict_raises(self):
        with self.assertRaises(OpenAICompatError):
            _parse_tool_arguments('[1, 2, 3]')

    def test_unsupported_type_raises(self):
        with self.assertRaises(OpenAICompatError):
            _parse_tool_arguments(12345)


class TestParseUsage(unittest.TestCase):
    def test_standard_fields(self):
        usage = _parse_usage({'input_tokens': 10, 'output_tokens': 20})
        self.assertEqual(usage.input_tokens, 10)
        self.assertEqual(usage.output_tokens, 20)

    def test_prompt_completion_aliases(self):
        usage = _parse_usage({'prompt_tokens': 15, 'completion_tokens': 25})
        self.assertEqual(usage.input_tokens, 15)
        self.assertEqual(usage.output_tokens, 25)

    def test_ollama_aliases(self):
        usage = _parse_usage({'prompt_eval_count': 12, 'eval_count': 18})
        self.assertEqual(usage.input_tokens, 12)
        self.assertEqual(usage.output_tokens, 18)

    def test_cache_tokens(self):
        usage = _parse_usage({
            'input_tokens': 1,
            'output_tokens': 1,
            'cache_creation_input_tokens': 100,
            'cache_read_input_tokens': 200,
        })
        self.assertEqual(usage.cache_creation_input_tokens, 100)
        self.assertEqual(usage.cache_read_input_tokens, 200)

    def test_reasoning_tokens_top_level_and_details(self):
        usage_top = _parse_usage({'input_tokens': 1, 'output_tokens': 1, 'reasoning_tokens': 50})
        self.assertEqual(usage_top.reasoning_tokens, 50)

        usage_details = _parse_usage({
            'input_tokens': 1,
            'output_tokens': 1,
            'completion_tokens_details': {'reasoning_tokens': 75},
        })
        self.assertEqual(usage_details.reasoning_tokens, 75)

    def test_non_dict_returns_empty(self):
        usage = _parse_usage('not a dict')
        self.assertEqual(usage, UsageStats())

    def test_string_number_coercion(self):
        usage = _parse_usage({'input_tokens': '10', 'output_tokens': '20'})
        self.assertEqual(usage.input_tokens, 10)
        self.assertEqual(usage.output_tokens, 20)


class TestBuildResponseFormat(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(_build_response_format(None))

    def test_valid_schema(self):
        schema = OutputSchemaConfig(
            name='test_schema',
            schema={'type': 'object', 'properties': {'x': {'type': 'integer'}}},
            strict=True,
        )
        result = _build_response_format(schema)
        self.assertEqual(result, {
            'type': 'json_schema',
            'json_schema': {
                'name': 'test_schema',
                'schema': {'type': 'object', 'properties': {'x': {'type': 'integer'}}},
                'strict': True,
            },
        })


class TestOptionalInt(unittest.TestCase):
    def test_int_passthrough(self):
        self.assertEqual(_optional_int(42), 42)

    def test_float_truncated(self):
        self.assertEqual(_optional_int(3.9), 3)

    def test_string_parsed(self):
        self.assertEqual(_optional_int('7'), 7)

    def test_bool_returns_zero(self):
        self.assertEqual(_optional_int(True), 0)
        self.assertEqual(_optional_int(False), 0)

    def test_none_returns_zero(self):
        self.assertEqual(_optional_int(None), 0)

    def test_invalid_string_returns_zero(self):
        self.assertEqual(_optional_int('abc'), 0)


if __name__ == '__main__':
    unittest.main()
