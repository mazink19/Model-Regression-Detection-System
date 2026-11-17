from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field


class ExpectedToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    matchers: dict[str, Literal["exact", "regex", "any", "one_of", "approx"]] = Field(default_factory=dict)


class OutcomeChecks(BaseModel):
    contains: list[str] = Field(default_factory=list)


class ExpectedBehavior(BaseModel):
    tool_calls: list[ExpectedToolCall] = Field(default_factory=list)
    unordered: bool = False
    forbidden_tools: list[str] = Field(default_factory=list)
    outcome: OutcomeChecks = Field(default_factory=OutcomeChecks)


class CaseLimits(BaseModel):
    max_steps: int = Field(default=20, gt=0)
    timeout_s: float = Field(default=60, gt=0)


class TestCase(BaseModel):
    __test__: ClassVar[bool] = False
    model_config = ConfigDict(extra="forbid")
    id: str
    tags: list[str] = Field(default_factory=list)
    input: dict[str, Any]
    expected: ExpectedBehavior = Field(default_factory=ExpectedBehavior)
    limits: CaseLimits = Field(default_factory=CaseLimits)


class ToolCall(BaseModel):
    kind: Literal["tool"] = "tool"
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None
    latency_ms: float = 0


class LLMCall(BaseModel):
    kind: Literal["llm"] = "llm"
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0
    cached: bool = False


TraceStep = ToolCall | LLMCall


class Trace(BaseModel):
    case_id: str
    steps: list[TraceStep] = Field(default_factory=list)
    final_answer: str = ""
    total_latency_ms: float = 0
    total_tokens: int = 0
    terminated_by: Literal["answer", "max_steps", "timeout", "error"] = "answer"
    error_type: str | None = None

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [step for step in self.steps if isinstance(step, ToolCall)]


class Score(BaseModel):
    metric: str
    value: float
    explanation: str = ""


class CaseEvaluation(BaseModel):
    case_id: str
    repeat_idx: int
    passed: bool
    scores: list[Score]
    failures: list[str] = Field(default_factory=list)
    trace: Trace


class AgentRun(BaseModel):
    id: str
    created_at: str
    agent_name: str
    cases: list[CaseEvaluation]
