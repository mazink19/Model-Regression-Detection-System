from agent_eval.adapters import FakeAgent
from agent_eval.dataset import load_dataset
from agent_eval.models import TestCase, Trace


def test_sample_dataset_loads_and_fake_agent_returns_trace():
    cases = load_dataset("dataset/agent_eval_sample.yaml")
    assert len(cases) == 30
    case = cases[0]
    assert case.expected.outcome.contains == ["Strong match"]

    trace = Trace(case_id=case.id, final_answer="Strong match", steps=[])
    assert FakeAgent({case.id: trace}).run(case) == trace


def test_career_dataset_balances_strong_weak_and_missing_information():
    cases = load_dataset("dataset/agent_eval_sample.yaml")
    ids = {case.id for case in cases}
    strong = [case for case in cases if "strong-match" in case.tags]
    weak = [case for case in cases if "weak-match" in case.tags]
    sparse = [case for case in cases if "missing-information" in case.tags]
    assert len(ids) == 30
    assert len(strong) == 15
    assert len(weak) == 15
    assert len(sparse) == 3


def test_dataset_rejects_invalid_cases(tmp_path):
    file = tmp_path / "bad.yaml"
    file.write_text("- id: missing-input\n", encoding="utf-8")
    try:
        load_dataset(file)
    except ValueError as exc:
        assert "missing-input" in str(exc)
    else:
        raise AssertionError("invalid case should be rejected")


def test_testcase_rejects_unknown_fields():
    try:
        TestCase.model_validate({"id": "x", "input": {}, "unexpected": True})
    except Exception:
        pass
    else:
        raise AssertionError("unknown fields should be rejected")
