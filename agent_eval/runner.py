from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timezone
from uuid import uuid4

from agent_eval.adapters import AgentAdapter
from agent_eval.models import AgentRun, CaseEvaluation, TestCase, Trace
from agent_eval.scoring import classify_failures, score_trace


def run_suite(cases: list[TestCase], adapter: AgentAdapter, repeats: int = 1, known_tools: set[str] | None = None) -> AgentRun:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    results = []
    for case in cases:
        for repeat_idx in range(repeats):
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(adapter.run, case)
            try:
                trace = future.result(timeout=case.limits.timeout_s)
            except TimeoutError:
                future.cancel()
                trace = Trace(case_id=case.id, terminated_by="timeout")
            except Exception as exc:
                trace = Trace(case_id=case.id, terminated_by="error",
                              final_answer=f"Adapter execution failed ({type(exc).__name__})",
                              error_type=type(exc).__name__)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            if trace.case_id != case.id:
                trace = trace.model_copy(update={"case_id": case.id})
            if len(trace.steps) > case.limits.max_steps:
                trace = trace.model_copy(update={
                    "steps": trace.steps[:case.limits.max_steps],
                    "terminated_by": "max_steps",
                })
            scores = score_trace(case, trace, known_tools)
            failures = classify_failures(case, trace, scores, known_tools)
            values = {score.metric: score.value for score in scores}
            passed = values.get("task_success") == 1 and values.get("tool_selection") == 1
            results.append(CaseEvaluation(case_id=case.id, repeat_idx=repeat_idx, passed=passed,
                                          scores=scores, failures=failures, trace=trace))
    return AgentRun(id=uuid4().hex, created_at=datetime.now(timezone.utc).isoformat(), agent_name=adapter.name, cases=results)
