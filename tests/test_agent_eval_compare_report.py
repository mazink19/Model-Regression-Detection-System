from agent_eval.compare import compare_runs
from agent_eval.models import AgentRun, CaseEvaluation, Score, Trace
from agent_eval.report import write_report


def _run(run_id, passed):
    return AgentRun(id=run_id, created_at="2026-01-01T00:00:00Z", agent_name="fake", cases=[
        CaseEvaluation(case_id="c", repeat_idx=0, passed=passed, trace=Trace(case_id="c"),
                       scores=[Score(metric="task_success", value=float(passed)), Score(metric="tool_selection", value=float(passed)), Score(metric="tool_args", value=float(passed))])
    ])


def test_gate_detects_degradation_and_report_contains_regression(tmp_path):
    baseline, current = _run("base", True), _run("current", False)
    result = compare_runs(baseline, current)
    assert result.passed is False
    assert result.regressions == ["c"]
    assert result.pass_at_k == 0
    assert result.pass_pow_k == 0
    files = write_report(baseline, current, result, tmp_path)
    assert "c" in files["html"].read_text(encoding="utf-8")
    assert files["json"].exists() and files["markdown"].exists()


def test_identical_run_passes_gate():
    assert compare_runs(_run("a", True), _run("b", True)).passed


def test_repeat_summary_reports_flaky_case():
    baseline = _run("base", True)
    current = _run("current", True)
    failed_repeat = current.cases[0].model_copy(update={
        "repeat_idx": 1,
        "passed": False,
        "scores": [Score(metric="task_success", value=0), Score(metric="tool_selection", value=1), Score(metric="tool_args", value=1)],
    })
    current.cases.append(failed_repeat)
    result = compare_runs(baseline, current)
    assert result.pass_at_k == 1
    assert result.pass_pow_k == 0
    assert result.flaky_cases == ["c"]
