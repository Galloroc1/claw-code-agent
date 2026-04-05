from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.agent_plugin_cache import (
    MAX_PLUGIN_LINES,
    MAX_PLUGIN_PREVIEW_CHARS,
    PluginCacheEntry,
    _coerce_entry,
    _extract_entries,
    discover_plugin_cache,
    load_plugin_cache_summary,
)


class TestDiscoverPluginCacheNone(unittest.TestCase):
    """discover_plugin_cache returns None when no cache files exist."""

    def test_returns_none_for_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = discover_plugin_cache(Path(tmp))
            self.assertIsNone(result)

    def test_returns_none_when_port_sessions_dir_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".port_sessions").mkdir()
            result = discover_plugin_cache(Path(tmp))
            self.assertIsNone(result)


class TestDiscoverPluginCacheListFormat(unittest.TestCase):
    """discover_plugin_cache finds cache in .port_sessions/plugin_cache.json (list format)."""

    def test_list_of_strings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / ".port_sessions"
            cache_dir.mkdir()
            cache_file = cache_dir / "plugin_cache.json"
            cache_file.write_text(json.dumps(["plugin-a", "plugin-b"]))

            result = discover_plugin_cache(Path(tmp))
            self.assertIsNotNone(result)
            self.assertIn("plugin-a", result)
            self.assertIn("plugin-b", result)
            self.assertIn("Plugin entries discovered: 2", result)

    def test_list_of_dicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / ".port_sessions"
            cache_dir.mkdir()
            cache_file = cache_dir / "plugin_cache.json"
            cache_file.write_text(
                json.dumps([{"name": "alpha", "version": "1.0"}, {"name": "beta"}])
            )

            result = discover_plugin_cache(Path(tmp))
            self.assertIsNotNone(result)
            self.assertIn("alpha", result)
            self.assertIn("version=1.0", result)
            self.assertIn("beta", result)


class TestDiscoverPluginCacheDictPluginsKey(unittest.TestCase):
    """discover_plugin_cache finds cache in .port_sessions/plugins.json (dict with 'plugins' key)."""

    def test_dict_with_plugins_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / ".port_sessions"
            cache_dir.mkdir()
            cache_file = cache_dir / "plugins.json"
            payload = {"plugins": [{"name": "foo"}, {"name": "bar"}]}
            cache_file.write_text(json.dumps(payload))

            result = discover_plugin_cache(Path(tmp))
            self.assertIsNotNone(result)
            self.assertIn("foo", result)
            self.assertIn("bar", result)
            self.assertIn("Plugin entries discovered: 2", result)


class TestDiscoverPluginCacheDictEntriesKey(unittest.TestCase):
    """discover_plugin_cache handles dict with 'entries' key format."""

    def test_dict_with_entries_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / ".port_sessions"
            cache_dir.mkdir()
            cache_file = cache_dir / "plugin_cache.json"
            payload = {"entries": [{"name": "entry-a"}, {"name": "entry-b"}]}
            cache_file.write_text(json.dumps(payload))

            result = discover_plugin_cache(Path(tmp))
            self.assertIsNotNone(result)
            self.assertIn("entry-a", result)
            self.assertIn("entry-b", result)


class TestDiscoverPluginCacheDictKeyAsName(unittest.TestCase):
    """discover_plugin_cache handles dict where values are dicts (key=name format)."""

    def test_dict_values_are_dicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / ".port_sessions"
            cache_dir.mkdir()
            cache_file = cache_dir / "plugin_cache.json"
            payload = {
                "my-plugin": {"version": "2.0", "source": "/path/to/it"},
                "other-plugin": {"version": "3.1"},
            }
            cache_file.write_text(json.dumps(payload))

            result = discover_plugin_cache(Path(tmp))
            self.assertIsNotNone(result)
            self.assertIn("my-plugin", result)
            self.assertIn("version=2.0", result)
            self.assertIn("source=/path/to/it", result)
            self.assertIn("other-plugin", result)


class TestCoerceEntry(unittest.TestCase):
    """_coerce_entry handles various input types."""

    def test_string_entry(self) -> None:
        entry = _coerce_entry("simple-plugin")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.name, "simple-plugin")
        self.assertTrue(entry.enabled)

    def test_string_entry_strips_whitespace(self) -> None:
        entry = _coerce_entry("  padded-name  ")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.name, "padded-name")

    def test_empty_string_returns_none(self) -> None:
        self.assertIsNone(_coerce_entry(""))
        self.assertIsNone(_coerce_entry("   "))

    def test_dict_with_name_key(self) -> None:
        entry = _coerce_entry({"name": "named-plugin"})
        self.assertIsNotNone(entry)
        self.assertEqual(entry.name, "named-plugin")

    def test_dict_with_plugin_key(self) -> None:
        entry = _coerce_entry({"plugin": "plugin-key"})
        self.assertIsNotNone(entry)
        self.assertEqual(entry.name, "plugin-key")

    def test_dict_with_id_key(self) -> None:
        entry = _coerce_entry({"id": "id-key"})
        self.assertIsNotNone(entry)
        self.assertEqual(entry.name, "id-key")

    def test_name_takes_precedence_over_plugin_and_id(self) -> None:
        entry = _coerce_entry({"name": "winner", "plugin": "loser", "id": "also-loser"})
        self.assertIsNotNone(entry)
        self.assertEqual(entry.name, "winner")

    def test_dict_with_version_and_source(self) -> None:
        entry = _coerce_entry(
            {"name": "full", "version": "1.2.3", "source": "/src"}
        )
        self.assertIsNotNone(entry)
        self.assertEqual(entry.version, "1.2.3")
        self.assertEqual(entry.source, "/src")

    def test_source_fallback_to_path(self) -> None:
        entry = _coerce_entry({"name": "p", "path": "/a/b"})
        self.assertIsNotNone(entry)
        self.assertEqual(entry.source, "/a/b")

    def test_source_fallback_to_module(self) -> None:
        entry = _coerce_entry({"name": "p", "module": "my.mod"})
        self.assertIsNotNone(entry)
        self.assertEqual(entry.source, "my.mod")

    def test_disabled_plugin(self) -> None:
        entry = _coerce_entry({"name": "off", "enabled": False})
        self.assertIsNotNone(entry)
        self.assertFalse(entry.enabled)

    def test_enabled_none_defaults_to_true(self) -> None:
        entry = _coerce_entry({"name": "on"})
        self.assertIsNotNone(entry)
        self.assertTrue(entry.enabled)

    def test_enabled_truthy_value(self) -> None:
        entry = _coerce_entry({"name": "on", "enabled": 1})
        self.assertIsNotNone(entry)
        self.assertTrue(entry.enabled)

    def test_empty_dict_returns_none(self) -> None:
        self.assertIsNone(_coerce_entry({}))

    def test_non_string_returns_none(self) -> None:
        self.assertIsNone(_coerce_entry(42))
        self.assertIsNone(_coerce_entry(None))
        self.assertIsNone(_coerce_entry(True))
        self.assertIsNone(_coerce_entry([]))

    def test_dict_with_non_string_name_returns_none(self) -> None:
        self.assertIsNone(_coerce_entry({"name": 123}))
        self.assertIsNone(_coerce_entry({"name": ""}))

    def test_empty_version_is_none(self) -> None:
        entry = _coerce_entry({"name": "p", "version": ""})
        self.assertIsNotNone(entry)
        self.assertIsNone(entry.version)

    def test_non_string_version_is_none(self) -> None:
        entry = _coerce_entry({"name": "p", "version": 5})
        self.assertIsNotNone(entry)
        self.assertIsNone(entry.version)

    def test_empty_source_is_none(self) -> None:
        entry = _coerce_entry({"name": "p", "source": ""})
        self.assertIsNotNone(entry)
        self.assertIsNone(entry.source)


class TestExtractEntries(unittest.TestCase):
    """_extract_entries handles all payload shapes."""

    def test_list_payload(self) -> None:
        entries = _extract_entries(["a", "b"])
        self.assertEqual(len(entries), 2)

    def test_dict_plugins_key(self) -> None:
        entries = _extract_entries({"plugins": [{"name": "x"}]})
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].name, "x")

    def test_dict_entries_key(self) -> None:
        entries = _extract_entries({"entries": [{"name": "y"}]})
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].name, "y")

    def test_dict_key_as_name(self) -> None:
        entries = _extract_entries({"k1": {"version": "1"}, "k2": {"version": "2"}})
        names = {e.name for e in entries}
        self.assertEqual(names, {"k1", "k2"})

    def test_plugins_key_takes_precedence_over_key_as_name(self) -> None:
        payload = {"plugins": [{"name": "from-plugins"}], "other": {"version": "1"}}
        entries = _extract_entries(payload)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].name, "from-plugins")

    def test_non_dict_values_ignored_in_key_as_name(self) -> None:
        entries = _extract_entries({"good": {"version": "1"}, "bad": "string-val"})
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].name, "good")

    def test_empty_list_returns_empty(self) -> None:
        self.assertEqual(_extract_entries([]), [])

    def test_invalid_payload_type(self) -> None:
        self.assertEqual(_extract_entries("not-valid"), [])
        self.assertEqual(_extract_entries(42), [])


class TestLoadPluginCacheSummary(unittest.TestCase):
    """load_plugin_cache_summary returns rendered summary string."""

    def test_returns_none_when_no_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = load_plugin_cache_summary(Path(tmp))
            self.assertIsNone(result)

    def test_returns_summary_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / ".port_sessions"
            cache_dir.mkdir()
            (cache_dir / "plugin_cache.json").write_text(
                json.dumps([{"name": "my-plugin", "version": "1.0"}])
            )

            result = load_plugin_cache_summary(Path(tmp))
            self.assertIsNotNone(result)
            self.assertIn("my-plugin", result)
            self.assertIn("Plugin cache loaded from:", result)


class TestRenderedSummaryCounts(unittest.TestCase):
    """Rendered summary shows correct enabled/disabled counts."""

    def _make_cache(self, tmp: str, entries: list) -> str | None:
        cache_dir = Path(tmp) / ".port_sessions"
        cache_dir.mkdir(exist_ok=True)
        (cache_dir / "plugin_cache.json").write_text(json.dumps(entries))
        return discover_plugin_cache(Path(tmp))

    def test_all_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._make_cache(tmp, [{"name": "a"}, {"name": "b"}, {"name": "c"}])
            self.assertIn("Enabled plugins: 3", result)
            self.assertNotIn("Disabled plugins:", result)

    def test_some_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._make_cache(
                tmp,
                [
                    {"name": "a"},
                    {"name": "b", "enabled": False},
                    {"name": "c", "enabled": False},
                ],
            )
            self.assertIn("Enabled plugins: 1", result)
            self.assertIn("Disabled plugins: 2", result)

    def test_disabled_shown_in_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._make_cache(
                tmp, [{"name": "off-plugin", "enabled": False}]
            )
            self.assertIn("disabled", result)
            self.assertIn("off-plugin", result)


class TestPreviewTruncation(unittest.TestCase):
    """Preview truncation works (MAX_PLUGIN_PREVIEW_CHARS=4000)."""

    def test_long_output_is_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / ".port_sessions"
            cache_dir.mkdir()
            # Create entries with very long names so the rendered output exceeds the limit
            long_name = "x" * 500
            entries = [{"name": f"{long_name}-{i}"} for i in range(20)]
            (cache_dir / "plugin_cache.json").write_text(json.dumps(entries))

            result = discover_plugin_cache(Path(tmp))
            self.assertIsNotNone(result)
            self.assertLessEqual(len(result), MAX_PLUGIN_PREVIEW_CHARS)
            self.assertTrue(result.endswith("..."))

    def test_short_output_not_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / ".port_sessions"
            cache_dir.mkdir()
            entries = [{"name": "small"}]
            (cache_dir / "plugin_cache.json").write_text(json.dumps(entries))

            result = discover_plugin_cache(Path(tmp))
            self.assertIsNotNone(result)
            self.assertFalse(result.endswith("..."))


class TestMaxPluginLinesTruncation(unittest.TestCase):
    """More than MAX_PLUGIN_LINES (12) shows truncation message."""

    def test_more_than_max_lines_shows_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / ".port_sessions"
            cache_dir.mkdir()
            count = MAX_PLUGIN_LINES + 5
            entries = [{"name": f"plugin-{i}"} for i in range(count)]
            (cache_dir / "plugin_cache.json").write_text(json.dumps(entries))

            result = discover_plugin_cache(Path(tmp))
            self.assertIsNotNone(result)
            self.assertIn(f"... plus 5 more plugin entries", result)
            self.assertIn(f"Plugin entries discovered: {count}", result)

    def test_exactly_max_lines_no_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / ".port_sessions"
            cache_dir.mkdir()
            entries = [{"name": f"plugin-{i}"} for i in range(MAX_PLUGIN_LINES)]
            (cache_dir / "plugin_cache.json").write_text(json.dumps(entries))

            result = discover_plugin_cache(Path(tmp))
            self.assertIsNotNone(result)
            self.assertNotIn("more plugin entries", result)


class TestMalformedJsonSkipped(unittest.TestCase):
    """Malformed JSON files are gracefully skipped."""

    def test_invalid_json_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / ".port_sessions"
            cache_dir.mkdir()
            (cache_dir / "plugin_cache.json").write_text("{not valid json!!!")

            result = discover_plugin_cache(Path(tmp))
            self.assertIsNone(result)

    def test_malformed_first_valid_second(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / ".port_sessions"
            cache_dir.mkdir()
            # First candidate: malformed
            (cache_dir / "plugin_cache.json").write_text("not json")
            # Second candidate: valid
            (cache_dir / "plugins.json").write_text(
                json.dumps({"plugins": [{"name": "fallback"}]})
            )

            result = discover_plugin_cache(Path(tmp))
            self.assertIsNotNone(result)
            self.assertIn("fallback", result)

    def test_valid_json_but_empty_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / ".port_sessions"
            cache_dir.mkdir()
            (cache_dir / "plugin_cache.json").write_text(json.dumps([]))

            result = discover_plugin_cache(Path(tmp))
            self.assertIsNone(result)


class TestAdditionalWorkingDirectories(unittest.TestCase):
    """additional_working_directories are searched."""

    def test_finds_cache_in_additional_dir(self) -> None:
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as extra:
            cache_dir = Path(extra) / ".port_sessions"
            cache_dir.mkdir()
            (cache_dir / "plugin_cache.json").write_text(
                json.dumps([{"name": "extra-plugin"}])
            )

            result = discover_plugin_cache(Path(main), (extra,))
            self.assertIsNotNone(result)
            self.assertIn("extra-plugin", result)

    def test_main_dir_preferred_over_additional(self) -> None:
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as extra:
            for base, name in [(main, "main-plugin"), (extra, "extra-plugin")]:
                cache_dir = Path(base) / ".port_sessions"
                cache_dir.mkdir()
                (cache_dir / "plugin_cache.json").write_text(
                    json.dumps([{"name": name}])
                )

            result = discover_plugin_cache(Path(main), (extra,))
            self.assertIsNotNone(result)
            self.assertIn("main-plugin", result)

    def test_load_plugin_cache_summary_with_additional_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as main, tempfile.TemporaryDirectory() as extra:
            cache_dir = Path(extra) / ".port_sessions"
            cache_dir.mkdir()
            (cache_dir / "plugin_cache.json").write_text(
                json.dumps([{"name": "via-summary"}])
            )

            result = load_plugin_cache_summary(Path(main), (extra,))
            self.assertIsNotNone(result)
            self.assertIn("via-summary", result)


class TestPluginCacheEntry(unittest.TestCase):
    """PluginCacheEntry dataclass behavior."""

    def test_defaults(self) -> None:
        entry = PluginCacheEntry(name="test")
        self.assertEqual(entry.name, "test")
        self.assertTrue(entry.enabled)
        self.assertIsNone(entry.version)
        self.assertIsNone(entry.source)

    def test_frozen(self) -> None:
        entry = PluginCacheEntry(name="test")
        with self.assertRaises(AttributeError):
            entry.name = "other"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
