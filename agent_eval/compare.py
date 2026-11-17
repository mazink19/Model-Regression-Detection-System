from collections import defaultdict

from pydantic import BaseModel

from agent_eval.models import AgentRun


class MetricDelta(BaseModel):
    metric: str
    baseline: float
    current: float
    delta: float


class RunComparison(BaseModel):
    passed: bool
    regressions: list[str]
    improvements: list[str]
    flaky_cases: list[str]
    pass_at_k: float
    pass_pow_k: float
    metrics: list[MetricDelta]


def compare_runs(baseline: AgentRun, current: AgentRun, max_drop: float = 0.03) -> RunComparison:
    def aggregates(run):
        values: dict[str, list[float]] = defaultdict(list)
        for case in run.cases:
            for score in case.scores:
                if score.metric not in {"latency_ms", "tokens"}:
                    values[score.metric].append(score.value)
        values["task_success_rate"] = [float(case.passed) for case in run.cases]
        return {key: sum(vals) / len(vals) for key, vals in values.items() if vals}

    before, after = aggregates(baseline), aggregates(current)
    deltas = [MetricDelta(metric=metric, baseline=value, current=after.get(metric, 0), delta=after.get(metric, 0) - value)
              for metric, value in sorted(before.items())]
    def pass_rates(run):
        grouped: dict[str, list[bool]] = defaultdict(list)
        for case in run.cases:
            grouped[case.case_id].append(case.passed)
        return {case_id: sum(outcomes) / len(outcomes) for case_id, outcomes in grouped.items()}

    old, new = pass_rates(baseline), pass_rates(current)
    shared_ids = old.keys() & new.keys()
    regressions = sorted(cid for cid in shared_ids if old[cid] == 1 and new[cid] == 0)
    improvements = sorted(cid for cid in shared_ids if old[cid] == 0 and new[cid] == 1)
    flaky_cases = sorted(cid for cid, rate in new.items() if 0 < rate < 1)
    grouped_current: dict[str, list[bool]] = defaultdict(list)
    for case in current.cases:
        grouped_current[case.case_id].append(case.passed)
    # Empirical pass@k is the share of cases with any successful repeat;
    # pass^k is the share whose every repeat succeeds.
    pass_at_k = sum(any(outcomes) for outcomes in grouped_current.values()) / len(grouped_current) if grouped_current else 0.0
    pass_pow_k = sum(all(outcomes) for outcomes in grouped_current.values()) / len(grouped_current) if grouped_current else 0.0
    gated = [d for d in deltas if d.metric in {"task_success_rate", "tool_selection", "tool_args"}]
    return RunComparison(passed=all(d.delta >= -max_drop for d in gated), regressions=regressions,
                         improvements=improvements, flaky_cases=flaky_cases,
                         pass_at_k=pass_at_k, pass_pow_k=pass_pow_k, metrics=deltas)
