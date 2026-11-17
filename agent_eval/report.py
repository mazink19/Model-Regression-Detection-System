import html
import json
from pathlib import Path

from agent_eval.compare import RunComparison
from agent_eval.models import AgentRun


def write_report(baseline: AgentRun, current: AgentRun, comparison: RunComparison, out: str | Path = "reports/agent-eval") -> dict[str, Path]:
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"baseline_run": baseline.id, "current_run": current.id,
               "passed": comparison.passed, "regressions": comparison.regressions,
               "improvements": comparison.improvements,
               "flaky_cases": comparison.flaky_cases,
               "pass_at_k": comparison.pass_at_k, "pass_pow_k": comparison.pass_pow_k,
               "metrics": [m.model_dump() for m in comparison.metrics],
               "cases": [c.model_dump(mode="json") for c in current.cases]}
    json_path = folder / "summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path = folder / "summary.md"
    md_path.write_text("# Agent evaluation\n\n" + f"**Gate:** {'PASS' if comparison.passed else 'FAIL'}\n\n" +
                       f"**pass@k:** {comparison.pass_at_k:.1%}  \n**pass^k:** {comparison.pass_pow_k:.1%}\n\n" +
                       "| Metric | Baseline | Current | Delta |\n|---|---:|---:|---:|\n" +
                       "".join(f"| {m.metric} | {m.baseline:.3f} | {m.current:.3f} | {m.delta:+.3f} |\n" for m in comparison.metrics) +
                       "\nRegressions: " + (", ".join(comparison.regressions) or "none") +
                       "\nFlaky cases: " + (", ".join(comparison.flaky_cases) or "none") + "\n", encoding="utf-8")
    rows = "".join(f"<tr><td>{html.escape(m.metric)}</td><td>{m.baseline:.3f}</td><td>{m.current:.3f}</td><td>{m.delta:+.3f}</td></tr>" for m in comparison.metrics)
    cases_html = "".join(
        f"<details><summary>{html.escape(case.case_id)} — {'PASS' if case.passed else 'FAIL'} — repeat {case.repeat_idx + 1}</summary>"
        f"<p>Failures: {html.escape(', '.join(case.failures) or 'none')}</p>"
        f"<pre>{html.escape(case.trace.model_dump_json(indent=2))}</pre></details>"
        for case in current.cases
    )
    html_path = folder / "report.html"
    html_path.write_text(f"<!doctype html><meta charset='utf-8'><title>Agent evaluation</title><style>body{{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse}}td,th{{padding:.5rem;border:1px solid #bbb}}pre{{white-space:pre-wrap;background:#f5f5f5;padding:1rem}}</style><h1>Agent evaluation: {'PASS' if comparison.passed else 'FAIL'}</h1><p>Baseline {html.escape(baseline.id)} → current {html.escape(current.id)}</p><p>pass@k: {comparison.pass_at_k:.1%}; pass^k: {comparison.pass_pow_k:.1%}</p><table><tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Delta</th></tr>{rows}</table><h2>Regressions</h2><ul>{''.join(f'<li>{html.escape(cid)}</li>' for cid in comparison.regressions) or '<li>None</li>'}</ul><h2>Improvements</h2><ul>{''.join(f'<li>{html.escape(cid)}</li>' for cid in comparison.improvements) or '<li>None</li>'}</ul><h2>Flaky cases</h2><ul>{''.join(f'<li>{html.escape(cid)}</li>' for cid in comparison.flaky_cases) or '<li>None</li>'}</ul><h2>Case traces</h2>{cases_html}", encoding="utf-8")
    return {"json": json_path, "markdown": md_path, "html": html_path}
