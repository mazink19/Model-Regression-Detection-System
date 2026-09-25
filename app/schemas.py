from dataclasses import dataclass
from typing import Literal
from pydantic import BaseModel, Field

@dataclass
class PromptConfig:
    version: str
    system_prompt: str

class ClassificationResult(BaseModel):
    category: Literal["billing", "technical", "general", "account"]
    summary: str


class EvaluationCaseResult(BaseModel):
    case_id: str
    expected_category: str
    actual_category: str
    expected_summary: str
    actual_summary: str
    passed: bool


class EvaluationRun(BaseModel):
    total_cases: int
    passed: int
    failed: int
    accuracy: float
    cases: list[EvaluationCaseResult]    

class CaseChange(BaseModel):
    case_id : str
    expected_category : str
    baseline_category: str
    current_category: str


class CategoryComparison(BaseModel):
    category: str
    baseline_accuracy: float
    current_accuracy: float
    accuracy_delta: float


class RegressionStatus(BaseModel):
    status: Literal["pass", "warning", "critical"]
    accuracy_delta: float
    regressions: list[CaseChange] = Field(default_factory=list)
    improvements: list[CaseChange] = Field(default_factory=list)
    category_comparisons : list[CategoryComparison] = Field(default_factory=list)