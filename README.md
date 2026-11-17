# Model Regression Detector

Model Regression Detector is a small evaluation harness for checking whether changes to an LLM application make its behavior worse. It runs examples from a dataset, scores the resulting traces, saves results, compares a run with a baseline, and produces reports that can be reviewed locally or in CI.

This repository currently contains two related pieces:

1. The original single-call evaluator under `app/`, which compares model answers against a golden dataset.
2. The newer `agent_eval/` harness, which evaluates tool-using agents and is integrated with Career Copilot's job-matching LangGraph.

The agent harness is the active architecture milestone. See [architecture.md](architecture.md) for the target design and its implementation status.

## What an evaluation does

```text
Dataset cases -> Agent or model -> Trace / answer -> Scorers -> Saved run
                                                               |
Baseline run <---------------- Comparison and regression gate <-+
                                  |
                                  +-> HTML, JSON, and Markdown reports
```

Each dataset case describes an input and the expected behavior. For agent cases, expectations can include the tools that should be called, argument values, ordering, and required answer content. Scorers turn the observed trace into pass/fail results and metrics. A comparison checks the current run against a named baseline; CI can fail when the comparison detects a regression.

## Requirements and setup

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for installing the pinned project dependencies
- For the Career Copilot integration: a local checkout of `mazink19/ai-career-copilot`, its backend dependencies, and provider configuration when making live model calls

From the repository root, install the harness dependencies:

```powershell
uv sync --frozen
```

Run the unit tests with:

```powershell
uv run python -m pytest -q
```

The harness and tests use deterministic fixtures; this command does not call an LLM provider.

## Run the standalone fake-agent example

The standalone CLI uses a deterministic fake agent. It reads `dataset/agent_eval_sample.yaml`, creates traces from the expected tool calls, and does not make network or model-provider requests.

```powershell
uv run python -m agent_eval.cli run --agent fake --dataset dataset/agent_eval_sample.yaml --repeats 1
```

The command prints a run ID and saves the run to `results/agent_eval.sqlite`. Use that ID to mark a baseline, compare a later run, enforce the gate, or generate reports:

```powershell
uv run python -m agent_eval.cli baseline-set --run <run-id> --name main
uv run python -m agent_eval.cli run --agent fake --dataset dataset/agent_eval_sample.yaml --repeats 3
uv run python -m agent_eval.cli compare --run latest --baseline main
uv run python -m agent_eval.cli gate --run latest --baseline main
uv run python -m agent_eval.cli report --run latest --baseline main --out reports/agent-eval
```

Replace `<run-id>` with the ID printed by the first command. `gate` exits with status 1 when the comparison fails. The CLI accepts `--db <path>` on these commands if you want to use a database other than `results/agent_eval.sqlite`.

The standalone fake adapter is useful for trying the runner, database, comparison, and reporting flow. It builds an expected trace from each case, so it is not a substitute for evaluating a real agent implementation.

## Evaluate Career Copilot

The integration invokes Career Copilot's actual job-matching graph. Run it from Career Copilot's backend directory so that its dependencies and local environment configuration are available. Set `MRDS_ROOT` to the absolute path of this repository first:

```powershell
$env:MRDS_ROOT = "E:\Model-Regression-Detection-System"
$env:PYTHONPATH = $env:MRDS_ROOT
python "$env:MRDS_ROOT\scripts\run_career_copilot_eval.py" --backend .
```

The script loads the 30 cases from `dataset/agent_eval_sample.yaml` and saves its run in this repository at `results/career-copilot.sqlite`. It prints the run ID, pass count, `pass@k`, `pass^k`, and any failures. `pass@k` means a case passed at least once across its repeats; `pass^k` means it passed every repeat.

Useful options:

| Option | Purpose |
|---|---|
| `--repeats N` | Run each case N times; helps reveal flaky behavior. |
| `--fake-matcher` | Use Career Copilot's deterministic test matcher rather than the live model matcher. |
| `--cache-mode record` | Save provider responses to the cache file for repeatable replay. |
| `--cache-mode replay` | Use only saved responses; a missing cache entry is a `cache_miss` and does not call the provider. |
| `--cache <path>` | Select the response-cache JSON file. |
| `--db <path>` | Select the SQLite run database. |

For example, to record live responses and then replay the same cases without making provider calls:

```powershell
python "$env:MRDS_ROOT\scripts\run_career_copilot_eval.py" --backend . --cache-mode record
python "$env:MRDS_ROOT\scripts\run_career_copilot_eval.py" --backend . --cache-mode replay
```

The checked-in replay cache is `baseline/career-copilot-cache.json`. Replay is deterministic against that cache, subject to the target application's graph and dependencies remaining compatible with the recorded responses. Keep credentials in Career Copilot's local environment configuration; do not put secrets in the dataset, cache, or Git.

## Current Career Copilot baseline

The named `main` baseline contains one recorded result for each of the 30 cases. The older five-repeat run is kept as `main-2case` for reference.

| Baseline | Cases | Repeats per case | Passing | pass@k | pass^k |
|---|---:|---:|---:|---:|---:|
| `main` | 30 | 1 | 30/30 | 100% | 100% |
| `main-2case` (archived) | 2 | 5 | 10/10 | 100% | 100% |

The latest full replay passed all 30 cases with cache hits and passed the comparison gate against `main`. Review the checked-in [evaluation summary](reports/agent-eval/summary.md) or [HTML report](reports/agent-eval/report.html). The cache and SQLite database provide the replay inputs and baseline history used by CI.

## GitHub Actions

The [agent evaluation workflow](.github/workflows/agent-eval.yml) runs on relevant pushes and pull requests targeting `main`, and can also be started manually from GitHub Actions. It checks out the Career Copilot backend, installs both projects, runs the harness tests, replays the 30 cases with the committed cache, compares the result with the `main` baseline, and uploads the generated report. For same-repository pull requests, it also updates a summary comment. A failed baseline gate fails the workflow.

The workflow reads the committed `baseline/career-copilot-cache.json` and `results/career-copilot.sqlite`. The replay is configured with placeholder environment variables and uses the cache, so CI does not need live provider credentials. It checks out Career Copilot's `main` branch; changes to that upstream branch can therefore affect future CI runs.

## Project layout

| Path | Contents |
|---|---|
| `app/` | Original answer-evaluation and regression-reporting implementation. |
| `agent_eval/` | Agent evaluation models, dataset loader, adapters, scorers, runner, cache, SQLite store, comparator, CLI, and report writer. |
| `dataset/` | Golden answer data and the agent evaluation YAML dataset. |
| `baseline/` | Original baseline and the Career Copilot replay cache. |
| `results/` | Evaluation run databases and original run artifacts. |
| `reports/` | Original report and generated agent evaluation reports. |
| `scripts/run_career_copilot_eval.py` | Entry point for running the Career Copilot graph against the agent dataset. |
| `tests/` | Unit and integration-adapter tests. |
| `.github/workflows/agent-eval.yml` | Automated replay, baseline gate, and report workflow. |
| `architecture.md` | Broader target architecture and milestone plan. |

## Current scope

Implemented agent-harness features include validated evaluation models, YAML case loading, a deterministic fake adapter, deterministic scoring and failure classification, repeat runs, a SQLite run store, named baselines, regression comparisons, static HTML/JSON/Markdown reports, LangGraph integration, and explicit LangChain response-cache record/replay. Reports include pass@k, pass^k, and mixed-repeat/flakiness information. The current Career Copilot dataset covers strong and weak job matches, including sparse resumes.

The architecture document describes the longer-term system. Some planned capabilities may not yet be implemented; consult its milestone status before treating the target design as a delivered feature set.
