from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    source: str
    excerpt: str = Field(min_length=1)


class Requirement(BaseModel):
    id: str
    text: str
    kind: Literal["functional", "constraint", "out_of_scope"] = "functional"
    evidence: Evidence


class TestStep(BaseModel):
    action: str = Field(min_length=1)
    expected_result: str = Field(min_length=1)


class TestCase(BaseModel):
    id: str
    title: str
    priority: Literal["P0", "P1", "P2"]
    type: Literal["functional", "negative", "regression"]
    preconditions: list[str]
    steps: list[TestStep] = Field(min_length=1)
    requirement_ids: list[str] = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)
    assumptions: list[str] = []


class UsageSummary(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    max_output_tokens: int | None = None


class RunLogEntry(BaseModel):
    timestamp: str
    agent: str
    level: Literal["INFO", "WARNING", "ERROR"] = "INFO"
    message: str

    @classmethod
    def create(cls, agent: str, message: str, level: Literal["INFO", "WARNING", "ERROR"] = "INFO") -> "RunLogEntry":
        return cls(timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"), agent=agent, level=level, message=message)


class TestSuite(BaseModel):
    prd_source: str
    requirements: list[Requirement]
    test_cases: list[TestCase]
    open_questions: list[str] = []
    usage: UsageSummary | None = None
    run_logs: list[RunLogEntry] = []
