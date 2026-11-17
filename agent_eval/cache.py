"""Explicit record/replay cache for JSON-serializable LLM responses."""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable


class LLMCache:
    def __init__(self, path: str | Path, mode: str = "replay"):
        if mode not in {"record", "replay"}:
            raise ValueError("cache mode must be 'record' or 'replay'")
        self.path = Path(path)
        self.mode = mode
        self.entries = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    @staticmethod
    def key(model: str, messages: Any, tools: Any, temperature: float) -> str:
        request = json.dumps({"model": model, "messages": messages, "tools": tools,
                              "temperature": temperature}, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
        return hashlib.sha256(request.encode("utf-8")).hexdigest()

    def call(self, model: str, messages: Any, tools: Any, temperature: float,
             request: Callable[[], Any] | None = None) -> Any:
        key = self.key(model, messages, tools, temperature)
        if self.mode == "replay":
            if key not in self.entries:
                raise KeyError(f"LLM replay cache miss for model {model!r} (key {key[:12]})")
            return self.entries[key]["response"]
        if request is None:
            raise ValueError("record mode requires a request callable")
        response = request()
        self.entries[key] = {"model": model, "messages": messages, "tools": tools,
                              "temperature": temperature, "response": response}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return response
