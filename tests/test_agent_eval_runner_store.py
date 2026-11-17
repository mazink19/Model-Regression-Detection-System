from agent_eval.adapters import FakeAgent
from agent_eval.models import TestCase, Trace, ToolCall
from agent_eval.runner import run_suite
from agent_eval.store import RunStore


def test_runner_repeats_and_sqlite_roundtrip(tmp_path):
    case = TestCase.model_validate({"id": "c", "input": {}, "expected": {"tool_calls": [{"name": "go"}], "outcome": {"contains": ["done"]}}})
    agent = FakeAgent({"c": Trace(case_id="c", steps=[ToolCall(name="go")], final_answer="done")})
    run = run_suite([case], agent, repeats=2)
    assert len(run.cases) == 2
    assert all(item.passed for item in run.cases)
    store = RunStore(tmp_path / "runs.sqlite")
    store.save(run)
    store.set_baseline("main", run.id)
    assert store.baseline("main").id == run.id


def test_run_suite_rejects_zero_repeats():
    try:
        run_suite([], FakeAgent({}), repeats=0)
    except ValueError as exc:
        assert "at least 1" in str(exc)
    else:
        raise AssertionError("zero repeats must be rejected")


def test_runner_enforces_max_steps():
    case = TestCase.model_validate({"id": "limited", "input": {}, "limits": {"max_steps": 1}})
    agent = FakeAgent({"limited": Trace(case_id="limited", steps=[ToolCall(name="a"), ToolCall(name="b")], final_answer="done")})
    result = run_suite([case], agent)
    assert len(result.cases[0].trace.steps) == 1
    assert result.cases[0].trace.terminated_by == "max_steps"
    assert "loop" in result.cases[0].failures
