"""All inter-agent data as Pydantic v2 models. Every agent boundary is validated."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SubQuestion(BaseModel):
    id: str
    text: str
    rationale: str = ""


class Plan(BaseModel):
    sub_questions: list[SubQuestion] = Field(min_length=3, max_length=5)

    @field_validator("sub_questions")
    @classmethod
    def unique_ids(cls, v: list[SubQuestion]) -> list[SubQuestion]:
        ids = [s.id for s in v]
        if len(set(ids)) != len(ids):
            raise ValueError("sub_question ids must be unique")
        return v


class Source(BaseModel):
    id: str
    url: str
    title: str = ""
    fetched_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Claim(BaseModel):
    text: str
    source_id: str
    supporting_quote: str = ""


class Findings(BaseModel):
    sub_question_id: str
    claims: list[Claim] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)


class ReportSection(BaseModel):
    heading: str
    body: str


class ReportHead(BaseModel):
    """Title + summary written in one call so sections can be written in parallel calls."""

    title: str
    summary: str


class Report(BaseModel):
    title: str
    summary: str
    sections: list[ReportSection]
    citations: list[str] = Field(default_factory=list)  # source ids referenced, e.g. ["S1", "S2"]


class Issue(BaseModel):
    type: Literal["unsupported_claim", "missing_coverage", "weak_source", "contradiction"]
    detail: str
    target: str = ""


class Critique(BaseModel):
    issues: list[Issue] = Field(default_factory=list)
    verdict: Literal["pass", "revise"] = "pass"


class CoverageJudgment(BaseModel):
    fact: str
    present: bool
    evidence: str = ""


class CoverageResult(BaseModel):
    judgments: list[CoverageJudgment]


class EntailmentJudgment(BaseModel):
    claim_index: int
    verdict: Literal["supported", "partially", "unsupported"]
    reason: str = ""


class EntailmentResult(BaseModel):
    judgments: list[EntailmentJudgment]


class BudgetState(BaseModel):
    llm_calls: int = 0
    tool_calls: int = 0
    tokens: int = 0
    started_at: float = 0.0
    reason: str = ""
