#!/usr/bin/env python3
"""
Download or export benchmark datasets into benchmarks/data.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from benchmarks.suites.aider import _BUILTIN_PROBLEMS as _AIDER_BUILTINS
from benchmarks.suites.aime import _BUILTIN_PROBLEMS as _AIME_BUILTINS
from benchmarks.suites.bfcl import _BUILTIN_PROBLEMS as _BFCL_BUILTINS
from benchmarks.suites.bigbench import _BUILTIN_PROBLEMS as _BIGBENCH_BUILTINS
from benchmarks.suites.codeforces import _BUILTIN_PROBLEMS as _CODEFORCES_BUILTINS
from benchmarks.suites.gpqa import _BUILTIN_PROBLEMS as _GPQA_BUILTINS
from benchmarks.suites.gsm8k import _BUILTIN_PROBLEMS as _GSM8K_BUILTINS
from benchmarks.suites.hle import _BUILTIN_PROBLEMS as _HLE_BUILTINS
from benchmarks.suites.humaneval import _BUILTIN_PROBLEMS as _HUMANEVAL_BUILTINS
from benchmarks.suites.ifeval import _BUILTIN_PROBLEMS as _IFEVAL_BUILTINS
from benchmarks.suites.livecodebench import _BUILTIN_PROBLEMS as _LIVECODEBENCH_BUILTINS
from benchmarks.suites.math_bench import _BUILTIN_PROBLEMS as _MATH_BUILTINS
from benchmarks.suites.mbpp import _BUILTIN_PROBLEMS as _MBPP_BUILTINS
from benchmarks.suites.mmmlu import _BUILTIN_PROBLEMS as _MMMLU_BUILTINS
from benchmarks.suites.mmlu_pro import _BUILTIN_PROBLEMS as _MMLU_PRO_BUILTINS
from benchmarks.suites.swe_bench import _BUILTIN_PROBLEMS as _SWE_BUILTINS
from benchmarks.suites.tau2 import _BUILTIN_PROBLEMS as _TAU2_BUILTINS


HF_DATASET_VIEWER_BASE = "https://datasets-server.huggingface.co"
HUMANEVAL_GZ_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"

JsonFetcher = Callable[[str, dict[str, object], dict[str, str] | None, float], object]


@dataclass
class DownloadResult:
    suite: str
    rows: int
    path: str
    source: str
    note: str = ""


def fetch_bytes(url: str, timeout: float, headers: dict[str, str] | None = None) -> bytes:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_json(
    endpoint: str,
    params: dict[str, object],
    headers: dict[str, str] | None,
    timeout: float,
) -> object:
    query = urllib.parse.urlencode(params, doseq=True)
    url = f"{HF_DATASET_VIEWER_BASE}/{endpoint}"
    if query:
        url = f"{url}?{query}"
    raw = fetch_bytes(url, timeout=timeout, headers=headers)
    return json.loads(raw.decode("utf-8"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return len(rows)


def _extract_gsm8k_answer(text: str) -> str:
    if "####" in text:
        text = text.split("####", 1)[1]
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    if numbers:
        return numbers[-1].replace(",", "")
    return text.strip().replace(",", "")


def _extract_math_answer(solution: str) -> str:
    boxed_fraction = re.search(r"\\boxed\{\\frac\{([^}]+)\}\{([^}]+)\}\}", solution, flags=re.DOTALL)
    if boxed_fraction:
        return f"{boxed_fraction.group(1).strip()}/{boxed_fraction.group(2).strip()}"
    boxed = re.search(r"\\boxed\{([^{}]+)\}", solution, flags=re.DOTALL)
    value = boxed.group(1) if boxed else solution
    value = value.strip()
    value = value.replace("\\frac{", "").replace("}{", "/").replace("}", "")
    value = value.replace("$", "").replace(",", "").strip()
    fraction = re.search(r"-?\d+\s*/\s*-?\d+", value)
    if fraction:
        return fraction.group(0).replace(" ", "")
    numbers = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", value)
    return numbers[-1] if numbers else value


def _fetch_hf_rows(
    dataset: str,
    *,
    config_preference: tuple[str, ...] = (),
    split_preference: tuple[str, ...] = ("test", "validation", "train"),
    json_fetcher: JsonFetcher = fetch_json,
    timeout: float = 60.0,
    headers: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    splits_payload = json_fetcher("splits", {"dataset": dataset}, headers, timeout)
    splits = list((splits_payload or {}).get("splits", []))  # type: ignore[assignment]
    if not splits:
        return []

    chosen: dict[str, Any] | None = None
    for config_name in config_preference:
        for split_name in split_preference:
            chosen = next(
                (
                    item for item in splits
                    if item.get("config") == config_name and item.get("split") == split_name
                ),
                None,
            )
            if chosen is not None:
                break
        if chosen is not None:
            break
    if chosen is None:
        for split_name in split_preference:
            chosen = next((item for item in splits if item.get("split") == split_name), None)
            if chosen is not None:
                break
    if chosen is None:
        chosen = splits[0]

    rows: list[dict[str, Any]] = []
    offset = 0
    length = 100
    while True:
        payload = json_fetcher(
            "rows",
            {
                "dataset": chosen["dataset"],
                "config": chosen["config"],
                "split": chosen["split"],
                "offset": offset,
                "length": length,
            },
            headers,
            timeout,
        )
        batch = [item.get("row", {}) for item in (payload or {}).get("rows", [])]  # type: ignore[union-attr]
        rows.extend(batch)
        total = int((payload or {}).get("num_rows_total", len(rows)))  # type: ignore[union-attr]
        offset += len(batch)
        if not batch or offset >= total:
            break
    return rows


def _download_humaneval(output_path: Path, *, timeout: float) -> DownloadResult:
    raw = fetch_bytes(HUMANEVAL_GZ_URL, timeout=timeout)
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    lines = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    rows = [
        {
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "canonical_solution": item.get("canonical_solution", ""),
            "test": item["test"],
            "entry_point": item["entry_point"],
        }
        for item in lines
    ]
    count = _write_jsonl(output_path, rows)
    return DownloadResult("humaneval", count, str(output_path), "official")


def _download_gsm8k(
    output_path: Path,
    *,
    timeout: float,
    json_fetcher: JsonFetcher = fetch_json,
) -> DownloadResult:
    rows = _fetch_hf_rows(
        "openai/gsm8k",
        config_preference=("main",),
        split_preference=("test",),
        json_fetcher=json_fetcher,
        timeout=timeout,
    )
    normalized = [
        {
            "id": f"gsm8k-{index + 1:04d}",
            "question": row["question"],
            "answer": _extract_gsm8k_answer(str(row["answer"])),
        }
        for index, row in enumerate(rows)
    ]
    count = _write_jsonl(output_path, normalized)
    return DownloadResult("gsm8k", count, str(output_path), "official")


def _download_mbpp(
    output_path: Path,
    *,
    timeout: float,
    json_fetcher: JsonFetcher = fetch_json,
) -> DownloadResult:
    rows = _fetch_hf_rows(
        "google-research-datasets/mbpp",
        config_preference=("sanitized", "full"),
        split_preference=("test", "validation"),
        json_fetcher=json_fetcher,
        timeout=timeout,
    )
    normalized = [
        {
            "task_id": row.get("task_id", index + 1),
            "text": row.get("text") or row.get("prompt") or "",
            "code": row.get("code", ""),
            "test_list": row.get("test_list") or row.get("test_setup_code", []),
        }
        for index, row in enumerate(rows)
    ]
    count = _write_jsonl(output_path, normalized)
    return DownloadResult("mbpp", count, str(output_path), "official")


def _download_math(
    output_path: Path,
    *,
    timeout: float,
    json_fetcher: JsonFetcher = fetch_json,
) -> DownloadResult:
    rows = _fetch_hf_rows(
        "hendrycks/competition_math",
        config_preference=("default",),
        split_preference=("test", "train"),
        json_fetcher=json_fetcher,
        timeout=timeout,
    )
    normalized = [
        {
            "id": row.get("problem_id", f"math-{index + 1:04d}"),
            "problem": row.get("problem", ""),
            "answer": _extract_math_answer(str(row.get("solution", row.get("answer", "")))),
            "subject": row.get("type", row.get("subject", "unknown")),
            "level": row.get("level", 0),
        }
        for index, row in enumerate(rows)
    ]
    count = _write_jsonl(output_path, normalized)
    return DownloadResult("math", count, str(output_path), "official")


def _download_mmlu_pro(
    output_path: Path,
    *,
    timeout: float,
    json_fetcher: JsonFetcher = fetch_json,
) -> DownloadResult:
    rows = _fetch_hf_rows(
        "TIGER-Lab/MMLU-Pro",
        config_preference=("default",),
        split_preference=("test", "validation"),
        json_fetcher=json_fetcher,
        timeout=timeout,
    )
    letters = "ABCDEFGHIJ"
    normalized = [
        {
            "id": f"mmlu-pro-{index + 1:04d}",
            "subject": row.get("category", row.get("subject", "unknown")),
            "question": row.get("question", ""),
            "choices": row.get("options", row.get("choices", [])),
            "answer": letters[row["answer"]] if isinstance(row.get("answer"), int) else str(row.get("answer", "")),
        }
        for index, row in enumerate(rows)
    ]
    count = _write_jsonl(output_path, normalized)
    return DownloadResult("mmlu-pro", count, str(output_path), "official")


def _download_gpqa(
    output_path: Path,
    *,
    timeout: float,
    json_fetcher: JsonFetcher = fetch_json,
) -> DownloadResult:
    rows = _fetch_hf_rows(
        "Idavidrein/gpqa",
        config_preference=("gpqa_diamond",),
        split_preference=("train",),
        json_fetcher=json_fetcher,
        timeout=timeout,
    )
    normalized = []
    for index, row in enumerate(rows):
        choices = [
            row.get("Correct Answer", ""),
            row.get("Incorrect Answer 1", ""),
            row.get("Incorrect Answer 2", ""),
            row.get("Incorrect Answer 3", ""),
        ]
        normalized.append({
            "id": f"gpqa-{index + 1:04d}",
            "subject": row.get("Subdomain", row.get("domain", "science")),
            "question": row.get("Question", ""),
            "choices": choices,
            "answer": "A",  # Correct answer is always first; shuffle at eval time if needed
        })
    count = _write_jsonl(output_path, normalized)
    return DownloadResult("gpqa-diamond", count, str(output_path), "official")


def _download_bigbench_hard(
    output_path: Path,
    *,
    timeout: float,
    json_fetcher: JsonFetcher = fetch_json,
) -> DownloadResult:
    rows = _fetch_hf_rows(
        "maveriq/bigbenchhard",
        config_preference=("default",),
        split_preference=("train",),
        json_fetcher=json_fetcher,
        timeout=timeout,
    )
    letters = "ABCDEFGHIJ"
    normalized = []
    for index, row in enumerate(rows):
        choices = row.get("choices", row.get("multiple_choice_targets", []))
        answer = row.get("answer", row.get("target", ""))
        if isinstance(answer, int) and answer < len(letters):
            answer = letters[answer]
        normalized.append({
            "id": f"bbh-{index + 1:04d}",
            "task": row.get("task", row.get("subject", "unknown")),
            "question": row.get("input", row.get("question", "")),
            "choices": choices,
            "answer": str(answer),
        })
    count = _write_jsonl(output_path, normalized)
    return DownloadResult("bigbench-hard", count, str(output_path), "official")


def _download_mmmlu(
    output_path: Path,
    *,
    timeout: float,
    json_fetcher: JsonFetcher = fetch_json,
) -> DownloadResult:
    rows = _fetch_hf_rows(
        "openai/MMMLU",
        config_preference=("default",),
        split_preference=("test", "validation"),
        json_fetcher=json_fetcher,
        timeout=timeout,
    )
    letters = "ABCD"
    normalized = [
        {
            "id": f"mmmlu-{index + 1:04d}",
            "language": row.get("language", "en"),
            "subject": row.get("subject", "unknown"),
            "question": row.get("question", ""),
            "choices": row.get("choices", row.get("options", [])),
            "answer": letters[row["answer"]] if isinstance(row.get("answer"), int) else str(row.get("answer", "")),
        }
        for index, row in enumerate(rows)
    ]
    count = _write_jsonl(output_path, normalized)
    return DownloadResult("mmmlu", count, str(output_path), "official")


def _download_hle(
    output_path: Path,
    *,
    timeout: float,
    json_fetcher: JsonFetcher = fetch_json,
) -> DownloadResult:
    rows = _fetch_hf_rows(
        "cais/hle",
        config_preference=("default",),
        split_preference=("test", "validation", "train"),
        json_fetcher=json_fetcher,
        timeout=timeout,
    )
    normalized = []
    for index, row in enumerate(rows):
        entry: dict[str, Any] = {
            "id": f"hle-{index + 1:04d}",
            "subject": row.get("category", row.get("subject", "general")),
            "question": row.get("question", ""),
            "answer": str(row.get("answer", "")),
        }
        if row.get("choices") or row.get("options"):
            entry["answer_type"] = "multiple_choice"
            entry["choices"] = row.get("choices", row.get("options", []))
        else:
            entry["answer_type"] = "exact"
        normalized.append(entry)
    count = _write_jsonl(output_path, normalized)
    return DownloadResult("hle", count, str(output_path), "official")


def _export_builtin(output_path: Path, suite: str, rows: list[dict[str, Any]], *, source: str = "builtin", note: str = "") -> DownloadResult:
    count = _write_jsonl(output_path, rows)
    return DownloadResult(suite, count, str(output_path), source, note)


def _builtin_rows(suite: str) -> list[dict[str, Any]]:
    mapping: dict[str, list[dict[str, Any]]] = {
        "humaneval": list(_HUMANEVAL_BUILTINS),
        "mbpp": list(_MBPP_BUILTINS),
        "gsm8k": list(_GSM8K_BUILTINS),
        "math": list(_MATH_BUILTINS),
        "swe-bench": list(_SWE_BUILTINS),
        "aider": list(_AIDER_BUILTINS),
        "livecodebench": list(_LIVECODEBENCH_BUILTINS),
        "aime": list(_AIME_BUILTINS),
        "ifeval": list(_IFEVAL_BUILTINS),
        "bfcl": list(_BFCL_BUILTINS),
        "mmlu-pro": list(_MMLU_PRO_BUILTINS),
        "gpqa-diamond": list(_GPQA_BUILTINS),
        "bigbench-hard": list(_BIGBENCH_BUILTINS),
        "mmmlu": list(_MMMLU_BUILTINS),
        "hle": list(_HLE_BUILTINS),
        "tau2": list(_TAU2_BUILTINS),
        "codeforces": list(_CODEFORCES_BUILTINS),
    }
    return list(mapping[suite])


def prepare_suite(
    suite: str,
    *,
    data_dir: Path,
    force: bool,
    builtin_only: bool,
    official_only: bool,
    timeout: float,
) -> DownloadResult:
    output_map = {
        "mmlu-pro": "mmlu_pro.jsonl",
        "gpqa-diamond": "gpqa.jsonl",
        "bigbench-hard": "bigbench_hard.jsonl",
        "mmmlu": "mmmlu.jsonl",
        "hle": "hle.jsonl",
        "tau2": "tau2.jsonl",
        "codeforces": "codeforces.jsonl",
    }
    output_path = data_dir / output_map.get(suite, f"{suite}.jsonl")
    if output_path.exists() and not force:
        lines = [line for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return DownloadResult(suite, len(lines), str(output_path), "existing")

    data_dir.mkdir(parents=True, exist_ok=True)

    builtin_only_suites = {"swe-bench", "aider", "livecodebench", "aime", "ifeval", "bfcl", "tau2", "codeforces"}
    official_downloaders = {
        "humaneval": _download_humaneval,
        "gsm8k": _download_gsm8k,
        "mbpp": _download_mbpp,
        "math": _download_math,
        "mmlu-pro": _download_mmlu_pro,
        "gpqa-diamond": _download_gpqa,
        "bigbench-hard": _download_bigbench_hard,
        "mmmlu": _download_mmmlu,
        "hle": _download_hle,
    }

    if suite in builtin_only_suites or builtin_only:
        return _export_builtin(output_path, suite, _builtin_rows(suite))

    downloader = official_downloaders.get(suite)
    if downloader is None:
        return _export_builtin(output_path, suite, _builtin_rows(suite))

    try:
        return downloader(output_path, timeout=timeout)
    except Exception as exc:
        if official_only:
            raise
        note = f"official download failed: {exc}"
        return _export_builtin(
            output_path,
            suite,
            _builtin_rows(suite),
            source="builtin-fallback",
            note=note,
        )


def _write_manifest(data_dir: Path, results: list[DownloadResult]) -> Path:
    manifest_path = data_dir / "manifest.json"
    payload = {
        "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [asdict(item) for item in results],
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download benchmark datasets for claw-code-agent.")
    parser.add_argument("--suite", action="append", default=[], help="Suite to prepare. Can be repeated.")
    parser.add_argument("--all", action="store_true", help="Prepare all known suites.")
    parser.add_argument("--list", action="store_true", help="List known suites.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Output data directory.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--builtin-only", action="store_true", help="Skip official downloads and export builtins only.")
    parser.add_argument("--official-only", action="store_true", help="Do not fall back to builtins.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Network timeout in seconds.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    known = [
        "humaneval",
        "mbpp",
        "gsm8k",
        "math",
        "swe-bench",
        "aider",
        "livecodebench",
        "aime",
        "ifeval",
        "bfcl",
        "mmlu-pro",
        "gpqa-diamond",
        "bigbench-hard",
        "mmmlu",
        "hle",
        "tau2",
        "codeforces",
    ]

    if args.list:
        for name in known:
            print(name)
        return

    suites = list(args.suite)
    if args.all:
        suites = known
    if not suites:
        parser.error("specify --suite or --all")

    results = [
        prepare_suite(
            suite,
            data_dir=args.data_dir,
            force=args.force,
            builtin_only=args.builtin_only,
            official_only=args.official_only,
            timeout=args.timeout,
        )
        for suite in suites
    ]
    manifest = _write_manifest(args.data_dir, results)
    print(f"Wrote {len(results)} suite files to {args.data_dir}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
