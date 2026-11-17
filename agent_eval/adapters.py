import importlib
import sys
import time
from pathlib import Path
from threading import RLock
from typing import Protocol

from agent_eval.models import TestCase, Trace


class AgentAdapter(Protocol):
    name: str

    def run(self, case: TestCase) -> Trace: ...


_LANGCHAIN_CACHE_LOCK = RLock()


class FakeAgent:
    """Returns a case's scripted trace; intended for harness tests and demos."""

    name = "fake"

    def __init__(self, traces: dict[str, Trace]):
        self.traces = traces

    def run(self, case: TestCase) -> Trace:
        if case.id not in self.traces:
            raise KeyError(f"FakeAgent has no scripted trace for case {case.id!r}")
        return self.traces[case.id].model_copy(deep=True)


class LangGraphAdapter:
    """Run a compiled LangGraph runnable and record LangChain callbacks.

    `input_mapper` can convert YAML-friendly input dictionaries into the
    target graph's Pydantic input objects. The graph is injected so this
    harness does not import or couple itself to one agent repository.
    """

    def __init__(self, graph, name: str = "langgraph-agent", input_mapper=None, llm_cache=None):
        self.graph = graph
        self.name = name
        self.input_mapper = input_mapper or (lambda case: case.input)
        self.llm_cache = llm_cache

    def run(self, case: TestCase) -> Trace:
        from agent_eval.langgraph_callbacks import TraceCallbackHandler

        handler = TraceCallbackHandler(cache=self.llm_cache)
        run_started = time.perf_counter()
        error_type = None
        try:
            inputs = self.input_mapper(case)
            if self.llm_cache is None:
                output = self.graph.invoke(inputs, config={"callbacks": [handler]})
            else:
                from langchain_core.globals import get_llm_cache, set_llm_cache

                with _LANGCHAIN_CACHE_LOCK:
                    previous_cache = get_llm_cache()
                    set_llm_cache(self.llm_cache)
                    try:
                        output = self.graph.invoke(inputs, config={"callbacks": [handler]})
                    finally:
                        set_llm_cache(previous_cache)
            answer = _final_answer(output)
            terminated_by = "answer"
        except Exception as exc:
            error_type = type(exc).__name__
            answer = f"Agent execution failed ({type(exc).__name__})"
            terminated_by = "error"
        steps, elapsed_ms, token_count = handler.finish()
        elapsed_ms = max(elapsed_ms, (time.perf_counter() - run_started) * 1000)
        return Trace(case_id=case.id, steps=steps, final_answer=answer,
                     total_latency_ms=elapsed_ms, total_tokens=token_count,
                     terminated_by=terminated_by, error_type=error_type)


def job_matching_input_mapper(resume_model):
    """Build the existing Career Copilot graph's ResumeAnalysis input shape."""
    def map_case(case: TestCase):
        inputs = dict(case.input)
        resume = inputs.get("resume")
        if isinstance(resume, dict):
            inputs["resume"] = resume_model.model_validate(resume)
        return inputs
    return map_case


def build_career_copilot_adapter(backend_path: str | Path, matcher=None, llm_cache=None) -> LangGraphAdapter:
    """Load the Career Copilot job-matching graph from its backend checkout.

    Call this from a fresh Python process with the target backend's runtime
    dependencies installed. A target `.env` is read relative to the current
    directory by that project's settings, so launch from its backend folder.
    """
    backend = Path(backend_path).resolve()
    if not (backend / "app" / "graph" / "job_matching" / "workflow.py").is_file():
        raise FileNotFoundError(f"Career Copilot backend not found at {backend}")
    loaded_app = sys.modules.get("app")
    if loaded_app is not None:
        raise RuntimeError("Load the Career Copilot adapter in a fresh process to avoid the target app package name collision")
    sys.path.insert(0, str(backend))
    try:
        workflow = importlib.import_module("app.graph.job_matching.workflow")
        matcher_module = importlib.import_module("app.ai.analyzers.job_matcher")
        resume_module = importlib.import_module("app.schemas.ai.resume_analysis")
        if matcher == "fake":
            fake_module = importlib.import_module("tests.fakes.fake_job_matcher")
            active_matcher = fake_module.FakeJobMatcher()
        elif matcher == "fake-mixed":
            fake_module = importlib.import_module("tests.fakes.fake_job_matcher")

            class BranchFakeMatcher:
                def match(self, resume, job_description):
                    description = job_description.casefold()
                    resume_skills = getattr(resume, "skills", []) or []
                    has_skill_overlap = any(skill.casefold() in description for skill in resume_skills)
                    delegate = fake_module.FakeJobMatcher() if has_skill_overlap else fake_module.WeakFakeJobMatcher()
                    return delegate.match(resume, job_description)

            active_matcher = BranchFakeMatcher()
        else:
            active_matcher = matcher if matcher is not None else matcher_module.JobMatcher()
        graph = workflow.build_job_matching_graph(active_matcher)
    except Exception:
        sys.path.remove(str(backend))
        raise
    return LangGraphAdapter(graph, name="career-copilot-job-matching",
                            input_mapper=job_matching_input_mapper(resume_module.ResumeAnalysis),
                            llm_cache=llm_cache)


def _final_answer(output) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        for key in ("final_answer", "answer", "recommendation", "output"):
            if output.get(key) is not None:
                value = output[key]
                analysis = output.get("analysis")
                return f"{value}\n{analysis}" if analysis is not None else str(value)
        return str(output)
    return str(output)
