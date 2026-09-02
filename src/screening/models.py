"""Pydantic models shared across the pipeline stages."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ResumeFile(BaseModel):
    """A resume file discovered during ingestion."""

    path: str
    filename: str
    content_hash: str


class ParsedResume(BaseModel):
    """Raw text successfully extracted from a resume file."""

    resume_file: ResumeFile
    text: str


class CandidateProfile(BaseModel):
    """Deterministically extracted candidate information."""

    name: str | None = None
    email: str | None = None
    phone: str | None = None
    github_url: str | None = None
    github_username: str | None = None
    skills_text: str = ""
    projects_text: str = ""
    raw_text: str = ""


class EligibilityResult(BaseModel):
    eligible: bool
    matched_skills: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    python_evidence: list[str] = Field(default_factory=list)
    ai_evidence_strong: list[str] = Field(default_factory=list)
    ai_evidence_weak: list[str] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    ai_project_depth: int = 0
    python_backend: int = 0
    cloud_fullstack: int = 0
    github: int = 0
    engineering_depth: int = 0

    @property
    def total(self) -> int:
        return (
            self.ai_project_depth
            + self.python_backend
            + self.cloud_fullstack
            + self.github
            + self.engineering_depth
        )


GithubStatus = Literal[
    "ok", "no_profile_found", "private_or_empty", "api_error", "rate_limited"
]


class GithubSummary(BaseModel):
    status: GithubStatus
    username: str | None = None
    recent_activity_score: int = 0
    maintained_repos_score: int = 0
    summary: str = ""
    reason: str | None = None

    @property
    def total_points(self) -> int:
        return self.recent_activity_score + self.maintained_repos_score


class AIJudgment(BaseModel):
    """Single structured LLM call output: AI project depth judgment + summary."""

    classification: Literal[
        "thin_wrapper", "tutorial_style", "applied_project", "production_grade_system"
    ]
    points: int = Field(ge=0, le=20)
    evidence: str
    reasoning: str
    project_summary: str
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)


class CandidateResult(BaseModel):
    candidate_name: str | None
    source_file: str
    email: str | None = None
    eligible: bool
    matched_skills: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    total_score: int | None = None
    score_breakdown: ScoreBreakdown | None = None
    project_summary: str | None = None
    github_summary: GithubSummary | None = None
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    rank: int | None = None


class FailedResume(BaseModel):
    source_file: str
    stage: str
    error: str


class BatchSummary(BaseModel):
    total_resumes: int
    successfully_parsed: int
    eligible: int
    rejected: int
    failed_unreadable: int
    generated_at: str


class ScreeningReport(BaseModel):
    batch_summary: BatchSummary
    eligible_candidates: list[CandidateResult] = Field(default_factory=list)
    rejected_candidates: list[CandidateResult] = Field(default_factory=list)
    failed_resumes: list[FailedResume] = Field(default_factory=list)
