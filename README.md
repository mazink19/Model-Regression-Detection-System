# Model Regression Detector

An automated evaluation and regression detection system for LLM-powered applications.

The system evaluates an LLM against a golden dataset, stores evaluation runs, compares a new run against a baseline, detects regressions, and generates an HTML evaluation report.

It also includes an early agent-evaluation harness for scripted tool-using agents. See [architecture.md](architecture.md) for the target design and [Agent Evaluation quickstart](#agent-evaluation-quickstart) for the implemented subset.

## Agent Evaluation quickstart

The harness validates YAML test cases, evaluates scripted traces, stores runs in SQLite, and compares a run with a named baseline. The fake adapter is deterministic and does not call an LLM.

```powershell
python -m agent_eval.cli run --agent fake --dataset dataset/agent_eval_sample.yaml
python -m agent_eval.cli baseline-set --run <run-id> --name main
python -m agent_eval.cli gate --run latest --baseline main
python -m agent_eval.cli report --run latest --baseline main --out reports/agent-eval
```

The current checkout implements the data model, YAML loader, fake adapter, deterministic scorers, failure classifier (including `rate_limit`, `cache_miss`, and `agent_error`), repeat runner, SQLite run store, baseline comparison/gate, static reports, and a callback-based LangGraph adapter with explicit record/replay caching. Reports include pass@k, pass^k, and mixed-repeat cases. `dataset/agent_eval_sample.yaml` has strong- and weak-match cases for Career Copilot's job-matching graph. Both graph branches passed against the target repository's fake matchers; this did not call the LLM.

### Career Copilot baseline

| Baseline | Cases | Repeats per case | Passing | pass@k | pass^k |
|---|---:|---:|---:|---:|---:|
| `main` | 30 | 1 | 30/30 | 100% | 100% |
| `main-2case` (archived) | 2 | 5 | 10/10 | 100% | 100% |

The `main` baseline is a live-model recording. The full 30-case replay then passed with 30/30 cache hits and passed the gate against `main`. The comparison report is in `reports/agent-eval/`; the replay cache is `baseline/career-copilot-cache.json`.

To run the real graph, use the Career Copilot backend's Python environment from its backend directory (so its `.env` is loaded), with this repository on `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "<MRDS_ROOT>"
python "<MRDS_ROOT>\scripts\run_career_copilot_eval.py" --backend .
```

That integration needs the Career Copilot backend dependencies and its configured LLM credentials. The harness still needs the cache/run artifacts committed and a PR CI workflow before it meets the full architecture.

To explicitly record LLM responses, add `--cache-mode record --cache <MRDS_ROOT>\baseline\career-copilot-cache.json`. Replay with the same dataset and cache path using `--cache-mode replay`; a cache miss is reported as `cache_miss` and does not fall through to a live provider call. The recorded 30-case cache is at `baseline/career-copilot-cache.json`; replay passed all 30 cases and the comparison gate passed against `main`.

The PR workflow is [.github/workflows/agent-eval.yml](.github/workflows/agent-eval.yml). It runs harness tests, replays the committed cache, gates against the SQLite `main` baseline, uploads the report, and updates a summary comment on same-repository pull requests. Commit the cache and baseline database for GitHub Actions to use them.

## Project Goal

LLM applications can change behavior when we modify:

- Prompts
- Models
- Model parameters
- Application logic

A change that improves one type of request can accidentally make another type worse.

This project provides an automated way to detect those regressions before they reach production.

## Architecture

```text
Golden Dataset
      |
      v
   Evaluator
      |
      v
 Evaluation Run
      |
      +----------------+
      |                |
      v                v
 baseline.json     current run
      |                |
      +-------+--------+
              |
              v
        Run Comparator
              |
      +-------+--------+
      |                |
      v                v
 Regressions      Improvements
      |
      v
Regression Status
      |
      v
   HTML Report
