"""Run the local dataset against the Career Copilot job-matching graph.

Launch this script from the Career Copilot backend directory so that its
settings load that checkout's .env and its dependencies are active.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent_eval.adapters import build_career_copilot_adapter
from agent_eval.dataset import load_dataset
from agent_eval.langchain_cache import LangChainResponseCache
from agent_eval.runner import run_suite
from agent_eval.store import RunStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=Path, default=Path.cwd())
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--fake-matcher", action="store_true", help="use Career Copilot's deterministic test matcher")
    parser.add_argument("--cache-mode", choices=["off", "record", "replay"], default="off")
    parser.add_argument("--cache", type=Path, default=PROJECT_ROOT / "baseline" / "career-copilot-cache.json")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "results" / "career-copilot.sqlite")
    parser.add_argument("--run-id-output", type=Path, help="append run_id=<id> to this file after saving the run")
    args = parser.parse_args()

    matcher = "fake-mixed" if args.fake_matcher else None
    cache = None if args.cache_mode == "off" else LangChainResponseCache(args.cache, args.cache_mode)
    adapter = build_career_copilot_adapter(args.backend, matcher=matcher, llm_cache=cache)
    cases = load_dataset(PROJECT_ROOT / "dataset" / "agent_eval_sample.yaml")
    run = run_suite(cases, adapter, repeats=args.repeats)
    RunStore(args.db).save(run)
    if args.run_id_output:
        with args.run_id_output.open("a", encoding="utf-8") as output:
            output.write(f"run_id={run.id}\n")
    passed = sum(case.passed for case in run.cases)
    by_case = defaultdict(list)
    for case in run.cases:
        by_case[case.case_id].append(case.passed)
    case_count = len(by_case)
    pass_at_k = sum(any(results) for results in by_case.values()) / case_count if case_count else 0
    pass_pow_k = sum(all(results) for results in by_case.values()) / case_count if case_count else 0
    print(f"Run {run.id}: {passed}/{len(run.cases)} repeats passing; "
          f"pass@{args.repeats}={pass_at_k:.0%}, pass^{args.repeats}={pass_pow_k:.0%}; saved to {args.db}")
    for case in run.cases:
        print(f"  {case.case_id}: {'PASS' if case.passed else 'FAIL'} {', '.join(case.failures)}")
    for case_id, outcomes in by_case.items():
        if any(outcomes) and not all(outcomes):
            print(f"  flaky: {case_id} ({sum(outcomes)}/{len(outcomes)} repeats passed)")


if __name__ == "__main__":
    main()
