from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from agent_eval.models import LLMCall, ToolCall


class TraceCallbackHandler(BaseCallbackHandler):
    """Capture model/tool callback events into harness trace steps."""

    def __init__(self, cache=None):
        self.steps: list[LLMCall | ToolCall] = []
        self.started: dict[str, float] = {}
        self.pending_tools: dict[str, tuple[str, dict[str, Any], float]] = {}
        self.model_names: dict[str, str | None] = {}
        self.total_latency_ms = 0.0
        self.total_tokens = 0
        self.cache = cache

    @staticmethod
    def _key(run_id) -> str:
        return str(run_id)

    def on_llm_start(self, serialized, prompts, run_id, **kwargs):
        key = self._key(run_id)
        self.started[key] = time.perf_counter()
        model_id = (serialized or {}).get("id")
        if isinstance(model_id, (list, tuple)):
            model_id = model_id[-1] if model_id else None
        self.model_names[key] = (serialized or {}).get("name") or model_id

    def on_chat_model_start(self, serialized, messages, run_id, **kwargs):
        self.on_llm_start(serialized, messages, run_id, **kwargs)

    def on_llm_end(self, response, run_id, **kwargs):
        key = self._key(run_id)
        start = self.started.pop(key, time.perf_counter())
        latency = max(0.0, (time.perf_counter() - start) * 1000)
        output = getattr(response, "llm_output", None) or {}
        usage = output.get("token_usage") or output.get("usage") or {}
        if not usage and getattr(response, "generations", None):
            message = response.generations[0][0].message
            usage = getattr(message, "usage_metadata", None) or {}
        self.total_latency_ms += latency
        cached = self.cache.consume_lookup_status() if self.cache is not None else False
        prompt = 0 if cached else (usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        completion = 0 if cached else (usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        self.total_tokens += int(prompt) + int(completion)
        self.steps.append(LLMCall(model=self.model_names.pop(key, None), prompt_tokens=int(prompt),
                                  completion_tokens=int(completion), latency_ms=latency, cached=cached))

    def on_llm_error(self, error, run_id, **kwargs):
        key = self._key(run_id)
        start = self.started.pop(key, time.perf_counter())
        latency = max(0.0, (time.perf_counter() - start) * 1000)
        self.total_latency_ms += latency
        self.model_names.pop(key, None)
        if self.cache is not None:
            self.cache.consume_lookup_status()

    def on_tool_start(self, serialized, input_str, run_id, **kwargs):
        key = self._key(run_id)
        name = (serialized or {}).get("name", "unknown")
        try:
            args = json.loads(input_str) if isinstance(input_str, str) else input_str
            if not isinstance(args, dict):
                args = {"input": args}
        except (TypeError, json.JSONDecodeError):
            args = {"input": str(input_str)}
        self.pending_tools[key] = (name, args, time.perf_counter())

    def on_tool_end(self, output, run_id, **kwargs):
        pending = self.pending_tools.pop(self._key(run_id), None)
        if pending is None:
            return
        name, args, start = pending
        latency = max(0.0, (time.perf_counter() - start) * 1000)
        self.total_latency_ms += latency
        self.steps.append(ToolCall(name=name, args=args, result=str(output), latency_ms=latency))

    def on_tool_error(self, error, run_id, **kwargs):
        pending = self.pending_tools.pop(self._key(run_id), None)
        if pending is None:
            return
        name, args, start = pending
        latency = max(0.0, (time.perf_counter() - start) * 1000)
        self.total_latency_ms += latency
        self.steps.append(ToolCall(name=name, args=args, error=str(error), latency_ms=latency))

    def finish(self):
        if self.cache is not None:
            self.cache.clear_pending_lookup_status()
        return self.steps, self.total_latency_ms, self.total_tokens
