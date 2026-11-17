from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent_eval.adapters import LangGraphAdapter
from agent_eval.langchain_cache import CacheMissError, LangChainResponseCache
from agent_eval.models import TestCase
from agent_eval.scoring import classify_failures, score_trace


class StubChatModel(BaseChatModel):
    model_name: str = "stub-model"
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "stub-chat-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="Stable career recommendation"))],
            llm_output={"token_usage": {"prompt_tokens": 5, "completion_tokens": 4}},
        )


class ModelGraph:
    def __init__(self, model):
        self.model = model

    def invoke(self, inputs, config):
        response = self.model.invoke(inputs["prompt"], config=config)
        return {"recommendation": response.content}


def test_langchain_cache_records_then_replays_without_model_call(tmp_path: Path):
    case = TestCase.model_validate({"id": "cache-case", "input": {"prompt": "hello"},
                                   "expected": {"outcome": {"contains": ["Stable career recommendation"]}}})
    path = tmp_path / "llm-cache.json"
    recording_model = StubChatModel()
    recording_adapter = LangGraphAdapter(ModelGraph(recording_model), llm_cache=LangChainResponseCache(path, "record"))
    recorded = recording_adapter.run(case)
    assert recorded.final_answer == "Stable career recommendation"
    assert recording_model.calls == 1
    assert recorded.steps[0].cached is False

    replay_model = StubChatModel()
    replay_adapter = LangGraphAdapter(ModelGraph(replay_model), llm_cache=LangChainResponseCache(path, "replay"))
    replayed = replay_adapter.run(case)
    assert replayed.final_answer == recorded.final_answer
    assert replay_model.calls == 0
    assert replayed.steps[0].cached is True
    assert replayed.steps[0].prompt_tokens == 0
    assert replayed.steps[0].completion_tokens == 0
    assert replayed.total_tokens == 0


def test_replay_cache_miss_never_calls_model(tmp_path: Path):
    case = TestCase.model_validate({"id": "cache-miss", "input": {"prompt": "not recorded"}})
    model = StubChatModel()
    adapter = LangGraphAdapter(ModelGraph(model), llm_cache=LangChainResponseCache(tmp_path / "empty.json", "replay"))
    trace = adapter.run(case)
    assert model.calls == 0
    assert trace.error_type == CacheMissError.__name__
    assert trace.terminated_by == "error"
    assert classify_failures(case, trace, score_trace(case, trace)) == ["cache_miss"]
