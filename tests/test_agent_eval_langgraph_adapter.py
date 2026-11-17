from types import SimpleNamespace
from uuid import uuid4

from agent_eval.adapters import LangGraphAdapter
from agent_eval.models import TestCase


class CallbackGraph:
    def invoke(self, inputs, config):
        assert inputs["prompt"] == "hello"
        handler = config["callbacks"][0]
        llm_id = uuid4()
        handler.on_chat_model_start({"id": ["provider", "model-x"]}, [], llm_id)
        handler.on_llm_end(SimpleNamespace(llm_output={"token_usage": {"prompt_tokens": 4, "completion_tokens": 3}}), llm_id)
        tool_id = uuid4()
        handler.on_tool_start({"name": "lookup"}, '{"id": 12}', tool_id)
        handler.on_tool_end({"ok": True}, tool_id)
        return {"recommendation": "Strong match"}


def test_langgraph_adapter_captures_callbacks_and_answer():
    case = TestCase.model_validate({"id": "c", "input": {"prompt": "hello"}})
    trace = LangGraphAdapter(CallbackGraph()).run(case)
    assert trace.terminated_by == "answer"
    assert "Strong match" in trace.final_answer
    assert [step.kind for step in trace.steps] == ["llm", "tool"]
    assert trace.steps[0].model == "model-x"
    assert trace.steps[0].prompt_tokens == 4
    assert trace.steps[1].name == "lookup"
    assert trace.steps[1].args == {"id": 12}
    assert trace.total_tokens == 7


def test_langgraph_adapter_records_execution_error():
    class BrokenGraph:
        def invoke(self, inputs, config):
            raise RuntimeError("graph failed")

    case = TestCase.model_validate({"id": "c", "input": {}})
    trace = LangGraphAdapter(BrokenGraph()).run(case)
    assert trace.terminated_by == "error"
    assert trace.error_type == "RuntimeError"
    assert "graph failed" not in trace.final_answer
