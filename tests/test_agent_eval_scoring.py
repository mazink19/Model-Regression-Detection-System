from agent_eval.models import TestCase, Trace, ToolCall
from agent_eval.scoring import classify_failures, score_trace


def test_scores_expected_calls_and_matchers():
    case = TestCase.model_validate({"id": "c", "input": {}, "expected": {"tool_calls": [
        {"name": "lookup", "args": {"id": 7, "kind": ["person", "org"]}, "matchers": {"id": "approx", "kind": "one_of"}}
    ], "outcome": {"contains": ["found"]}}})
    trace = Trace(case_id="c", steps=[ToolCall(name="lookup", args={"id": 7.004, "kind": "person"})], final_answer="Found it")
    scores = score_trace(case, trace, {"lookup"})
    assert {s.metric: s.value for s in scores}["tool_selection"] == 1
    assert {s.metric: s.value for s in scores}["task_success"] == 1
    assert classify_failures(case, trace, scores, {"lookup"}) == []


def test_classifies_common_tool_failures():
    case = TestCase.model_validate({"id": "c", "input": {}, "expected": {"tool_calls": [{"name": "lookup", "args": {"id": 1}}]}})
    trace = Trace(case_id="c", steps=[ToolCall(name="mystery", args={}), ToolCall(name="lookup", args={"id": 2})], final_answer="wrong")
    scores = score_trace(case, trace, {"lookup"})
    assert set(classify_failures(case, trace, scores, {"lookup"})) == {"wrong_tool", "bad_args", "wrong_order", "hallucinated_tool"}


def test_classifies_loop_timeout_and_tool_error():
    case = TestCase.model_validate({"id": "c", "input": {}, "expected": {"tool_calls": [
        {"name": "x"}, {"name": "x"}, {"name": "x"}
    ]}})
    trace = Trace(case_id="c", terminated_by="timeout", steps=[ToolCall(name="x", error="oops") for _ in range(3)])
    scores = score_trace(case, trace)
    assert set(classify_failures(case, trace, scores)) == {"loop", "unhandled_tool_error", "timeout"}


def test_rate_limit_is_not_misreported_as_wrong_answer():
    case = TestCase.model_validate({"id": "c", "input": {}, "expected": {"outcome": {"contains": ["done"]}}})
    trace = Trace(case_id="c", terminated_by="error", error_type="RateLimitError",
                  final_answer="Agent execution failed (RateLimitError)")
    scores = score_trace(case, trace)
    assert {score.metric: score.value for score in scores}["task_success"] == 0
    assert classify_failures(case, trace, scores) == ["rate_limit"]


def test_legacy_rate_limit_trace_is_classified_from_prefix():
    case = TestCase.model_validate({"id": "c", "input": {}, "expected": {"outcome": {"contains": ["done"]}}})
    trace = Trace(case_id="c", terminated_by="error", final_answer="RateLimitError: request throttled")
    assert classify_failures(case, trace, score_trace(case, trace)) == ["rate_limit"]
