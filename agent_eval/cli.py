import argparse

from agent_eval.adapters import FakeAgent
from agent_eval.compare import compare_runs
from agent_eval.dataset import load_dataset
from agent_eval.models import Trace, ToolCall
from agent_eval.report import write_report
from agent_eval.runner import run_suite
from agent_eval.store import RunStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent-eval")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="evaluate dataset with deterministic fake adapter")
    run.add_argument("--agent", choices=["fake"], default="fake")
    run.add_argument("--dataset", default="dataset/agent_eval_sample.yaml")
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--db", default="results/agent_eval.sqlite")
    base = sub.add_parser("baseline-set", help="set a named baseline")
    base.add_argument("--run", required=True)
    base.add_argument("--name", required=True)
    base.add_argument("--db", default="results/agent_eval.sqlite")
    for command in ("compare", "report", "gate"):
        item = sub.add_parser(command)
        item.add_argument("--run", default="latest")
        item.add_argument("--baseline", default="main")
        item.add_argument("--db", default="results/agent_eval.sqlite")
        item.add_argument("--out", default="reports/agent-eval")
    args = parser.parse_args()
    store = RunStore(args.db)
    if args.command == "baseline-set":
        store.set_baseline(args.name, args.run)
        print(f"Baseline {args.name!r} set to {args.run}")
        return
    if args.command in {"compare", "report", "gate"}:
        current = store.latest() if args.run == "latest" else store.load(args.run)
        baseline = store.baseline(args.baseline)
        comparison = compare_runs(baseline, current)
        if args.command == "report":
            print("\n".join(str(path) for path in write_report(baseline, current, comparison, args.out).values()))
        else:
            print(f"{'PASS' if comparison.passed else 'FAIL'}: {len(comparison.regressions)} regressions, {len(comparison.improvements)} improvements")
        if args.command == "gate" and not comparison.passed:
            raise SystemExit(1)
        return
    cases = load_dataset(args.dataset)
    scripts = {}
    for case in cases:
        steps = [ToolCall(name=call.name, args=call.args) for call in case.expected.tool_calls]
        answer = " ".join(case.expected.outcome.contains) or "Deterministic fake answer"
        scripts[case.id] = Trace(case_id=case.id, steps=steps, final_answer=answer)
    adapter = FakeAgent(scripts)
    result = run_suite(cases, adapter, args.repeats)
    store.save(result)
    total = len(result.cases)
    passed = sum(case.passed for case in result.cases)
    print(f"Run {result.id}: {passed}/{total} passing ({result.agent_name}); saved to {store.path}")


if __name__ == "__main__":
    main()
