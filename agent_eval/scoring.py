from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from agent_eval.models import Score, TestCase, Trace


def _matches(actual: Any, expected: Any, matcher: str) -> bool:
    if matcher == "any":
        return True
    if matcher == "regex":
        return isinstance(actual, str) and re.fullmatch(str(expected), actual) is not None
    if matcher == "one_of":
        return isinstance(expected, list) and actual in expected
    if matcher == "approx":
        if isinstance(expected, dict):
            target, tolerance = expected.get("value"), expected.get("tolerance", 0)
        else:
            target, tolerance = expected, 0.01
        try:
            return abs(float(actual) - float(target)) <= float(tolerance)
        except (TypeError, ValueError):
            return False
    return actual == expected


def score_trace(case: TestCase, trace: Trace, known_tools: set[str] | None = None) -> list[Score]:
    calls = trace.tool_calls
    expected = case.expected.tool_calls
    actual_names = [call.name for call in calls]
    expected_names = [call.name for call in expected]

    expected_counter, actual_counter = Counter(expected_names), Counter(actual_names)
    matched_count = sum(min(expected_counter[name], actual_counter[name]) for name in expected_counter)
    selection = 1.0 if actual_counter == expected_counter else 0.0
    if any(name in actual_counter for name in case.expected.forbidden_tools):
        selection = 0.0
    arg_checks = []
    consumed: Counter[str] = Counter()
    for want in expected:
        matches = [c for c in calls if c.name == want.name]
        idx = consumed[want.name]
        consumed[want.name] += 1
        if idx >= len(matches):
            arg_checks.append(False)
            continue
        call = matches[idx]
        fields_match = all(
            key in call.args and _matches(call.args[key], value, want.matchers.get(key, "exact"))
            for key, value in want.args.items()
        )
        exact_only = all(mode == "exact" for mode in want.matchers.values())
        if exact_only:
            fields_match = fields_match and set(call.args) == set(want.args)
        arg_checks.append(fields_match)
    args_score = sum(arg_checks) / len(arg_checks) if arg_checks else 1.0
    order = 1.0 if case.expected.unordered or actual_names == expected_names else 0.0
    registry_score = 1.0 if known_tools is None or all(name in known_tools for name in actual_names) else 0.0
    answer = trace.final_answer.casefold()
    outcome_checks = [phrase.casefold() in answer for phrase in case.expected.outcome.contains]
    if trace.terminated_by != "answer":
        task_success = 0.0
    else:
        task_success = sum(outcome_checks) / len(outcome_checks) if outcome_checks else float(bool(trace.final_answer.strip()))
    efficiency = min(1.0, len(expected) / len(calls)) if calls else float(not expected)
    return [
        Score(metric="tool_selection", value=selection, explanation=f"Matched {matched_count}/{sum(expected_counter.values())} expected calls; no forbidden calls"),
        Score(metric="tool_args", value=args_score, explanation=f"Arguments matched for {sum(arg_checks)}/{len(arg_checks)} expected calls"),
        Score(metric="tool_order", value=order, explanation="Expected call ordering" if order else "Tool calls were out of order"),
        Score(metric="no_hallucinated_tool", value=registry_score, explanation="All calls are registered" if registry_score else "Unknown tool called"),
        Score(metric="task_success", value=task_success, explanation="Outcome text checks"),
        Score(metric="step_efficiency", value=efficiency, explanation=f"{len(calls)} tool steps for {len(expected)} expected"),
        Score(metric="latency_ms", value=trace.total_latency_ms, explanation="Total trace latency"),
        Score(metric="tokens", value=float(trace.total_tokens), explanation="Total token count"),
    ]


def classify_failures(case: TestCase, trace: Trace, scores: list[Score], known_tools: set[str] | None = None) -> list[str]:
    by_metric = {score.metric: score.value for score in scores}
    got = [call.name for call in trace.tool_calls]
    want = [call.name for call in case.expected.tool_calls]
    failures: list[str] = []
    error_type = trace.error_type
    if error_type is None and trace.terminated_by == "error":
        legacy_prefix = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):", trace.final_answer)
        wrapper_type = re.search(r"\(([A-Za-z_][A-Za-z0-9_]*)\)", trace.final_answer)
        error_type = legacy_prefix.group(1) if legacy_prefix else (wrapper_type.group(1) if wrapper_type else None)
    if trace.terminated_by == "error":
        if error_type in {"RateLimitError", "TooManyRequests", "RateLimitExceeded"}:
            failures.append("rate_limit")
        elif error_type == "CacheMissError":
            failures.append("cache_miss")
        else:
            failures.append("agent_error")
    if any(name in got and name not in want for name in got) or any(name in got for name in case.expected.forbidden_tools):
        failures.append("wrong_tool")
    if any(got.count(name) < want.count(name) for name in set(want)):
        failures.append("missing_tool_call")
    if by_metric.get("tool_args", 1) < 1:
        failures.append("bad_args")
    if by_metric.get("tool_order", 1) < 1:
        failures.append("wrong_order")
    if known_tools is not None and any(name not in known_tools for name in got):
        failures.append("hallucinated_tool")
    repeated = Counter((c.name, json.dumps(c.args, sort_keys=True, default=str)) for c in trace.tool_calls)
    if trace.terminated_by == "max_steps" or any(count >= 3 for count in repeated.values()):
        failures.append("loop")
    if any(call.error for call in trace.tool_calls) and not any(call.error and call.error.casefold() in trace.final_answer.casefold() for call in trace.tool_calls):
        failures.append("unhandled_tool_error")
    if trace.terminated_by == "timeout":
        failures.append("timeout")
    if by_metric.get("task_success", 1) < 1 and not failures:
        failures.append("wrong_answer")
    elif by_metric.get("task_success", 1) < 1 and not any(f in failures for f in ("wrong_tool", "missing_tool_call", "bad_args", "wrong_order", "hallucinated_tool", "loop", "unhandled_tool_error", "timeout", "agent_error", "rate_limit", "cache_miss")):
        failures.append("wrong_answer")
    return failures
