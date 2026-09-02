"""The one LLM call site in the pipeline: AI project-depth judgment."""

from __future__ import annotations

from screening.llm.adapter import call_structured
from screening.llm.schemas import (
    AI_JUDGMENT_SYSTEM_PROMPT,
    AIJudgment,
    build_ai_judgment_prompt,
)
from screening.models import CandidateProfile


def judge_ai_project(profile: CandidateProfile) -> AIJudgment | None:
    prompt = build_ai_judgment_prompt(profile.raw_text)
    return call_structured(prompt, AIJudgment, system=AI_JUDGMENT_SYSTEM_PROMPT)
