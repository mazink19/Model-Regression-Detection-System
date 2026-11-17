import pytest

from agent_eval.cache import LLMCache


def test_record_then_replay_and_request_key_changes(tmp_path):
    path = tmp_path / "responses.json"
    recorder = LLMCache(path, "record")
    response = recorder.call("model-a", [{"role": "user", "content": "hi"}], [], 0, lambda: {"text": "hello"})
    assert response == {"text": "hello"}
    replay = LLMCache(path, "replay")
    assert replay.call("model-a", [{"role": "user", "content": "hi"}], [], 0) == response
    with pytest.raises(KeyError, match="cache miss"):
        replay.call("model-a", [{"role": "user", "content": "changed"}], [], 0)
