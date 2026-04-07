# Benchmarks — claw-code-agent

This directory contains two benchmark systems:

1. **Local task benchmarks** (`benchmarks/run.py`) — custom tasks that test agent capabilities directly
2. **Standard evaluation suites** (`benchmarks/run_suite.py`) — implementations of well-known AI evaluation benchmarks

---

## Quick Start

```bash
# From the repository root:

# Set up your model endpoint
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-4"                        # or your model name
export OPENAI_BASE_URL="http://localhost:8000/v1"   # if using local vLLM/ollama

# Run a quick smoke test (5 problems from HumanEval)
python3 -m benchmarks.run_suite --suite humaneval --limit 5 -v

# Run all benchmarks
python3 -m benchmarks.run_suite --all -o results.json
```

---

## Standard Evaluation Suites

### Available Benchmarks

| Suite | Category | # Problems (built-in) | Description |
|-------|----------|----------------------|-------------|
| **HumanEval** | Coding | 20 | Code generation from Python docstrings |
| **MBPP** | Coding | 15 | Basic Python programming problems |
| **SWE-Bench** | Coding | 5 | Resolve real-world GitHub issues |
| **Aider** | Coding | 6 | Code editing and refactoring tasks |
| **LiveCodeBench** | Coding | 5 | Competitive programming problems |
| **Codeforces** | Coding | 10 | Competitive programming with ELO rating |
| **MATH** | Math | 15 | Competition mathematics problems |
| **GSM8K** | Math | 15 | Grade school math word problems |
| **AIME** | Math | 10 | Challenging competition math (integers 0–999) |
| **IFEval** | Instruction Following | 10 | Verifiable instruction-following evaluation |
| **BFCL** | Instruction Following | 7 | Function/tool calling evaluation |
| **MMLU-Pro** | Knowledge | 10 | Professional-level multiple-choice QA (10 choices) |
| **GPQA-Diamond** | Knowledge | 10 | Graduate-level science QA (diamond difficulty) |
| **MMMLU** | Knowledge | 10 | Multilingual MMLU across languages |
| **HLE** | Knowledge | 10 | Humanity's Last Exam (extremely hard) |
| **BigBench-Hard** | Reasoning | 10 | BIG-Bench Extra Hard reasoning tasks |
| **Tau2** | Reasoning | 10 | Tool-augmented reasoning (retail/airline/finance) |

### Commands

#### List All Suites

```bash
python3 -m benchmarks.run_suite --list
```

#### Run a Specific Suite

```bash
# Coding benchmarks
python3 -m benchmarks.run_suite --suite humaneval
python3 -m benchmarks.run_suite --suite mbpp
python3 -m benchmarks.run_suite --suite swe-bench
python3 -m benchmarks.run_suite --suite aider
python3 -m benchmarks.run_suite --suite livecodebench
python3 -m benchmarks.run_suite --suite codeforces

# Math benchmarks
python3 -m benchmarks.run_suite --suite math
python3 -m benchmarks.run_suite --suite gsm8k
python3 -m benchmarks.run_suite --suite aime

# Knowledge benchmarks
python3 -m benchmarks.run_suite --suite mmlu-pro
python3 -m benchmarks.run_suite --suite gpqa-diamond
python3 -m benchmarks.run_suite --suite mmmlu
python3 -m benchmarks.run_suite --suite hle

# Reasoning benchmarks
python3 -m benchmarks.run_suite --suite bigbench-hard
python3 -m benchmarks.run_suite --suite tau2

# Instruction following benchmarks
python3 -m benchmarks.run_suite --suite ifeval
python3 -m benchmarks.run_suite --suite bfcl
```

#### Run by Category

```bash
# All coding benchmarks
python3 -m benchmarks.run_suite --category coding

# All math benchmarks
python3 -m benchmarks.run_suite --category math

# All knowledge benchmarks (MMLU-Pro, GPQA, MMMLU, HLE)
python3 -m benchmarks.run_suite --category knowledge

# All reasoning benchmarks (BigBench-Hard, Tau2)
python3 -m benchmarks.run_suite --category reasoning

# All instruction following benchmarks
python3 -m benchmarks.run_suite --category instruction-following
```

#### Run ALL Suites

```bash
python3 -m benchmarks.run_suite --all
```

#### Limit Problems (Quick Testing)

```bash
# Run only first 3 problems from each suite
python3 -m benchmarks.run_suite --all --limit 3

# Quick HumanEval smoke test
python3 -m benchmarks.run_suite --suite humaneval --limit 5 -v
```

#### Save Results to JSON

```bash
python3 -m benchmarks.run_suite --all -o results.json
python3 -m benchmarks.run_suite --suite humaneval -o humaneval_results.json
```

#### Verbose Output

```bash
python3 -m benchmarks.run_suite --suite humaneval -v
```

#### Custom Timeout

```bash
# 10 minutes per problem
python3 -m benchmarks.run_suite --suite swe-bench --timeout 600
```

#### Use Full Datasets (JSONL)

```bash
# Download or place your datasets in benchmarks/data/
python3 -m benchmarks.run_suite --suite humaneval --data-dir ./benchmarks/data/

# Or specify a custom directory
python3 -m benchmarks.run_suite --suite humaneval --data-dir /path/to/datasets/
```

---

## Using Full Datasets

Each suite looks for a JSONL file in the data directory. If the file isn't found, it falls back to the built-in subset.

| Suite | Expected File | Format |
|-------|--------------|--------|
| HumanEval | `humaneval.jsonl` | `{"task_id", "prompt", "canonical_solution", "test", "entry_point"}` |
| MBPP | `mbpp.jsonl` | `{"task_id", "text", "code", "test_list"}` |
| SWE-Bench | `swe_bench.jsonl` | `{"instance_id", "problem_statement", "setup_code", "test_cmd"}` |
| Aider | `aider.jsonl` | `{"id", "instruction", "setup_code", "test_code"}` |
| LiveCodeBench | `livecodebench.jsonl` | `{"id", "title", "description", "test_cases", "function_name"}` |
| Codeforces | `codeforces.jsonl` | `{"id", "rating", "title", "problem", "test_cases"}` |
| MATH | `math.jsonl` | `{"id", "problem", "answer", "subject", "level"}` |
| GSM8K | `gsm8k.jsonl` | `{"id", "question", "answer"}` |
| AIME | `aime.jsonl` or `aime_2026.jsonl` | `{"id", "problem", "answer"}` |
| IFEval | `ifeval.jsonl` | `{"id", "instruction", "checks"}` |
| BFCL | `bfcl.jsonl` | `{"id", "instruction", "expected_function", "setup_code", "test_code"}` |
| MMLU-Pro | `mmlu_pro.jsonl` | `{"id", "subject", "question", "choices", "answer"}` |
| GPQA-Diamond | `gpqa.jsonl` | `{"id", "subject", "question", "choices", "answer"}` |
| MMMLU | `mmmlu.jsonl` | `{"id", "language", "subject", "question", "choices", "answer"}` |
| HLE | `hle.jsonl` | `{"id", "subject", "question", "answer", "answer_type"}` |
| BigBench-Hard | `bigbench_hard.jsonl` | `{"id", "task", "question", "choices", "answer"}` |
| Tau2 | `tau2.jsonl` | `{"id", "domain", "question", "choices", "answer"}` |

### Downloading Full Datasets

```bash
# Download all datasets (builtin + official where available)
python3 -m benchmarks.download_datasets --all

# Download builtin-only (no network, uses embedded problems)
python3 -m benchmarks.download_datasets --all --builtin-only

# Download specific suites
python3 -m benchmarks.download_datasets --suite humaneval --suite mmlu-pro --suite gpqa-diamond

# Force re-download
python3 -m benchmarks.download_datasets --all --force
```

**Datasets with HuggingFace downloaders:**
HumanEval, GSM8K, MBPP, MATH, MMLU-Pro, GPQA-Diamond, BigBench-Hard, MMMLU, HLE

**Builtin-only datasets (no HuggingFace source):**
SWE-Bench, Aider, LiveCodeBench, AIME, IFEval, BFCL, Tau2, Codeforces

---

## Local Task Benchmarks

The original local benchmark system tests the agent on custom tasks.

```bash
# Run all local tasks
python3 -m benchmarks.run

# Run a single task
python3 -m benchmarks.run --task file-create-basic

# Run a category
python3 -m benchmarks.run --category bugfix

# Run by difficulty
python3 -m benchmarks.run --difficulty easy

# List available tasks
python3 -m benchmarks.run --list

# Verbose + save results
python3 -m benchmarks.run -v -o local_results.json
```

---

## All Commands Reference

```bash
# ─── STANDARD EVALUATION SUITES ────────────────────────────────────────

# List suites
python3 -m benchmarks.run_suite --list

# Individual suites
python3 -m benchmarks.run_suite --suite humaneval          # Coding: docstring → code
python3 -m benchmarks.run_suite --suite mbpp               # Coding: basic Python
python3 -m benchmarks.run_suite --suite swe-bench          # Coding: GitHub issues
python3 -m benchmarks.run_suite --suite aider              # Coding: code editing
python3 -m benchmarks.run_suite --suite livecodebench      # Coding: competitive programming
python3 -m benchmarks.run_suite --suite codeforces         # Coding: competitive + ELO
python3 -m benchmarks.run_suite --suite math               # Math: competition math
python3 -m benchmarks.run_suite --suite gsm8k              # Math: grade school
python3 -m benchmarks.run_suite --suite aime               # Math: AIME competition
python3 -m benchmarks.run_suite --suite mmlu-pro           # Knowledge: 14 subjects, 10 choices
python3 -m benchmarks.run_suite --suite gpqa-diamond       # Knowledge: graduate science
python3 -m benchmarks.run_suite --suite mmmlu              # Knowledge: multilingual MMLU
python3 -m benchmarks.run_suite --suite hle                # Knowledge: Humanity's Last Exam
python3 -m benchmarks.run_suite --suite bigbench-hard      # Reasoning: BIG-Bench Hard
python3 -m benchmarks.run_suite --suite tau2               # Reasoning: tool-augmented
python3 -m benchmarks.run_suite --suite ifeval             # Instruction: format following
python3 -m benchmarks.run_suite --suite bfcl               # Instruction: function calling

# Category runs
python3 -m benchmarks.run_suite --category coding                  # All coding
python3 -m benchmarks.run_suite --category math                    # All math
python3 -m benchmarks.run_suite --category knowledge               # MMLU-Pro, GPQA, MMMLU, HLE
python3 -m benchmarks.run_suite --category reasoning               # BigBench, Tau2
python3 -m benchmarks.run_suite --category instruction-following   # IFEval, BFCL

# Full run
python3 -m benchmarks.run_suite --all                              # All suites (~108 problems)
python3 -m benchmarks.run_suite --all --limit 3                    # Quick: 3 per suite
python3 -m benchmarks.run_suite --all -v -o results.json           # Full + verbose + save

# Options
python3 -m benchmarks.run_suite --suite humaneval --limit 5        # Limit problems
python3 -m benchmarks.run_suite --suite humaneval --timeout 600    # Custom timeout (seconds)
python3 -m benchmarks.run_suite --suite humaneval --data-dir ./data # Custom data directory
python3 -m benchmarks.run_suite --suite humaneval -v               # Verbose mode
python3 -m benchmarks.run_suite --suite humaneval -o out.json      # Save to JSON

# ─── LOCAL TASK BENCHMARKS ─────────────────────────────────────────────

python3 -m benchmarks.run --list                            # List local tasks
python3 -m benchmarks.run                                   # Run all local tasks
python3 -m benchmarks.run --task file-create-basic          # Single task
python3 -m benchmarks.run --category bugfix                 # Category filter
python3 -m benchmarks.run --difficulty easy                 # Difficulty filter
python3 -m benchmarks.run -v -o local_results.json          # Verbose + save
```

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | API key for the model provider | `sk-...` |
| `OPENAI_MODEL` | Model name to use | `gpt-4`, `qwen2.5-coder-32b` |
| `OPENAI_BASE_URL` | API base URL (for local models) | `http://localhost:8000/v1` |

---

## Output Format

Results are saved as JSON with this structure:

```json
{
  "benchmark_run": "claw-code-agent-evaluation",
  "timestamp": "2025-01-15T10:30:00",
  "suites": [
    {
      "suite_name": "HumanEval",
      "total": 20,
      "passed": 15,
      "failed": 5,
      "score_pct": 75.0,
      "duration_sec": 450.5,
      "model": "gpt-4",
      "results": [
        {
          "problem_id": "HumanEval/0",
          "passed": true,
          "duration_sec": 12.3,
          "error": ""
        }
      ]
    }
  ],
  "summary": {
    "total_suites": 10,
    "total_problems": 108,
    "total_passed": 85,
    "overall_score_pct": 78.7
  }
}
```

---

## Architecture

```
benchmarks/
├── __init__.py
├── run.py                  # Local task benchmark runner
├── run_suite.py            # Standard evaluation suite runner (CLI)
├── download_datasets.py    # Dataset downloader (HuggingFace + builtins)
├── README.md               # This file
├── data/                   # Dataset files (JSONL) — not committed
│   └── .gitkeep
├── tasks/
│   ├── __init__.py
│   └── definitions.py      # Local task definitions
└── suites/
    ├── __init__.py
    ├── base.py             # Base class for all suites
    ├── humaneval.py        # HumanEval benchmark
    ├── mbpp.py             # MBPP benchmark
    ├── swe_bench.py        # SWE-Bench benchmark
    ├── aider.py            # Aider benchmark
    ├── livecodebench.py    # LiveCodeBench benchmark
    ├── codeforces.py       # Codeforces (ELO scoring)
    ├── math_bench.py       # MATH benchmark
    ├── gsm8k.py            # GSM8K benchmark
    ├── aime.py             # AIME benchmark
    ├── mmlu_pro.py         # MMLU-Pro (10-choice QA)
    ├── gpqa.py             # GPQA Diamond (science)
    ├── mmmlu.py            # MMMLU (multilingual)
    ├── hle.py              # HLE (Humanity's Last Exam)
    ├── bigbench.py         # BigBench Extra Hard
    ├── tau2.py             # Tau2 (tool-augmented)
    ├── ifeval.py           # IFEval benchmark
    └── bfcl.py             # BFCL benchmark
```
