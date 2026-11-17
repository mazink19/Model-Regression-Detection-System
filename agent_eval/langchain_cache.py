"""Persistent, explicit record/replay cache for LangChain generations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from threading import Lock

from langchain_core.caches import BaseCache
from langchain_core.messages import message_to_dict, messages_from_dict
from langchain_core.outputs import ChatGeneration, Generation


class CacheMissError(RuntimeError):
    """Raised in replay mode instead of silently calling a live model."""


class LangChainResponseCache(BaseCache):
    def __init__(self, path: str | Path, mode: str):
        if mode not in {"record", "replay"}:
            raise ValueError("cache mode must be 'record' or 'replay'")
        self.path = Path(path)
        self.mode = mode
        self._lock = Lock()
        self._pending_lookup_status: list[bool] = []
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Unable to read LLM cache file {self.path}: {exc}") from exc
            if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("entries"), dict):
                raise ValueError(f"Unsupported or malformed LLM cache file: {self.path}")
            self.entries: dict[str, list[dict]] = data["entries"]
        else:
            self.entries = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(prompt: str, llm_string: str) -> str:
        request = json.dumps([prompt, llm_string], ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(request.encode("utf-8")).hexdigest()

    def lookup(self, prompt: str, llm_string: str) -> list[Generation] | None:
        key = self._key(prompt, llm_string)
        with self._lock:
            encoded = self.entries.get(key)
            self._pending_lookup_status.append(encoded is not None)
            if encoded is None:
                self.misses += 1
                if self.mode == "replay":
                    raise CacheMissError(f"LLM replay cache miss (key {key[:12]})")
                return None
            self.hits += 1
        try:
            decoded = [self._decode_generation(item) for item in encoded]
        except Exception as exc:
            raise ValueError(f"Invalid serialized generation for cache key {key[:12]}") from exc
        return decoded

    def update(self, prompt: str, llm_string: str, return_val: list[Generation]) -> None:
        if self.mode != "record":
            raise RuntimeError("Replay cache is read-only")
        key = self._key(prompt, llm_string)
        encoded = [self._encode_generation(item) for item in return_val]
        json.dumps(encoded, ensure_ascii=False)
        with self._lock:
            self.entries[key] = encoded
            self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"version": 1, "entries": self.entries}, indent=2, ensure_ascii=False)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent,
                                             prefix=f"{self.path.name}.", suffix=".tmp", delete=False) as temp:
                temp_path = Path(temp.name)
                temp.write(payload)
                temp.flush()
                os.fsync(temp.fileno())
            temp_path.replace(self.path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    @staticmethod
    def _encode_generation(generation: Generation) -> dict:
        if isinstance(generation, ChatGeneration):
            return {"kind": "chat", "message": message_to_dict(generation.message),
                    "generation_info": generation.generation_info}
        if isinstance(generation, Generation):
            return {"kind": "text", "text": generation.text,
                    "generation_info": generation.generation_info}
        raise TypeError(f"Unsupported LangChain generation type: {type(generation).__name__}")

    @staticmethod
    def _decode_generation(value: dict) -> Generation:
        if value.get("kind") == "chat":
            message = messages_from_dict([value["message"]])[0]
            return ChatGeneration(message=message, generation_info=value.get("generation_info"))
        if value.get("kind") == "text":
            return Generation(text=value["text"], generation_info=value.get("generation_info"))
        raise ValueError("Unknown generation kind")

    def clear(self, **kwargs) -> None:
        if self.mode != "record":
            raise RuntimeError("Replay cache is read-only")
        with self._lock:
            self.entries.clear()
            self.hits = 0
            self.misses = 0
            self._pending_lookup_status.clear()
            self._write()

    def consume_lookup_status(self) -> bool:
        with self._lock:
            return self._pending_lookup_status.pop(0) if self._pending_lookup_status else False

    def clear_pending_lookup_status(self) -> None:
        with self._lock:
            self._pending_lookup_status.clear()
