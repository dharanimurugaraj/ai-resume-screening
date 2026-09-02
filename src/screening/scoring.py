"""Stage 5: candidate scoring, 100-point rubric.

Everything here is deterministic except the AI-project-depth judgment, which
is optionally supplied by a single upstream LLM call (see llm/schemas.py
AIJudgment). Ai depth is split into two deterministic-friendly halves so the
system still produces a complete, reproducible score when the LLM is
unavailable: `ai_deterministic_score` (0-20, keyword/context based) plus
either the LLM's `points` (0-20) or a scaled fallback of the deterministic
half when no judgment is available.

"Reward evidence from projects over keyword-only skill lists" is implemented
literally: every weighted term is checked in `projects_text` first (full
weight) and only falls back to a smaller weight if it appears solely in
`skills_text`.
"""

from __future__ import annotations

import re

from screening.config import settings
from screening.eligibility import STRONG_AI_PATTERNS
from screening.models import AIJudgment, CandidateProfile, ScoreBreakdown

_WEIGHTS = settings.weights


def _term_present(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


def _weighted_score(
    terms: list[tuple[str, int]],
    profile: CandidateProfile,
    cap: int,
    half_weight_ratio: float = 0.4,
) -> int:
    """Sum `full_weight` for each term found in projects_text, or
    `full_weight * half_weight_ratio` if only found in skills_text, capped.
    """
    total = 0.0
    for pattern, full_weight in terms:
        if _term_present(pattern, profile.projects_text):
            total += full_weight
        elif _term_present(pattern, profile.skills_text):
            total += full_weight * half_weight_ratio
    return min(cap, round(total))


_PYTHON_BACKEND_TERMS: list[tuple[str, int]] = [
    (r"\bpython\b", 8),
    (r"\bfastapi\b", 6),
    (r"\bflask\b", 3),
    (r"\bdjango\b", 3),
    (r"\basync(io)?\b|\bawait\b", 4),
    (r"\bpostgres(ql)?\b", 4),
    (r"\bredis\b", 3),
]

_CLOUD_FULLSTACK_TERMS: list[tuple[str, int]] = [
    (r"\bdocker\b", 5),
    (r"\bkubernetes\b|\bk8s\b", 3),
    (r"\bgcp\b|\bgoogle\s+cloud\b|\baws\b|\bazure\b", 5),
    (r"\bci\s?/\s?cd\b|\bcontinuous\s+(integration|deployment)\b", 3),
    (r"\breact(\.js)?\b|\bnext\.js\b", 2),
    (r"\bdeployed\b|\bhosted\s+on\b|\blive\s+link\b|\bproduction\b", 2),
]

_ENGINEERING_DEPTH_TERMS: list[tuple[str, int]] = [
    (r"\b(unit|integration)\s+test(s|ing)?\b|\bpytest\b", 1),
    (r"\bcach(e|ing)\b", 1),
    (r"\bqueue(s|ing)?\b|\bcelery\b|\bkafka\b|\brabbitmq\b", 1),
    (r"\bobservability\b|\blogging\b|\bmonitoring\b|\bgrafana\b|\bprometheus\b", 1),
    (r"\bconcurren(cy|t)\b|\bmulti[\s-]?thread(ing|ed)?\b|\basyncio\b", 1),
    (r"\berror\s+handling\b|\bfailure\s+handling\b|\bretry\b|\bfault[\s-]toleran", 1),
]

_AI_DETERMINISTIC_TERMS: list[tuple[str, int]] = [(p, 4) for p in STRONG_AI_PATTERNS]


def score_python_backend(profile: CandidateProfile) -> int:
    return _weighted_score(_PYTHON_BACKEND_TERMS, profile, cap=_WEIGHTS.python_backend_max)


def score_cloud_fullstack(profile: CandidateProfile) -> int:
    return _weighted_score(_CLOUD_FULLSTACK_TERMS, profile, cap=_WEIGHTS.cloud_fullstack_max)


def score_engineering_depth(profile: CandidateProfile) -> int:
    return _weighted_score(
        _ENGINEERING_DEPTH_TERMS, profile, cap=_WEIGHTS.engineering_depth_max
    )


def score_ai_deterministic_base(profile: CandidateProfile) -> int:
    """0-20 pts from distinct strong AI-term hits, weighted toward project text."""
    return _weighted_score(
        _AI_DETERMINISTIC_TERMS,
        profile,
        cap=_WEIGHTS.ai_deterministic_max,
        half_weight_ratio=0.5,
    )


_THIN_CLASSIFICATIONS = {"thin_wrapper", "tutorial_style"}


def score_ai_project_depth(
    ai_deterministic_base: int, judgment: AIJudgment | None
) -> int:
    """Combine the deterministic base with the (optional) LLM judgment.

    Deterministic safety net: regardless of what the LLM returns, a
    thin-wrapper/tutorial-style classification is capped at 5 LLM points so
    one overly generous model call can't push a shallow project near the top
    of the ranking.
    """
    if judgment is None:
        # No LLM available: scale the 0-20 deterministic base up to the
        # full 0-40 range rather than leaving half the category empty.
        return min(_WEIGHTS.ai_project_depth_max, ai_deterministic_base * 2)

    llm_points = judgment.points
    if judgment.classification in _THIN_CLASSIFICATIONS:
        llm_points = min(llm_points, 5)

    return min(_WEIGHTS.ai_project_depth_max, ai_deterministic_base + llm_points)


def build_score_breakdown(
    profile: CandidateProfile,
    github_points: int,
    judgment: AIJudgment | None,
) -> ScoreBreakdown:
    ai_base = score_ai_deterministic_base(profile)
    return ScoreBreakdown(
        ai_project_depth=score_ai_project_depth(ai_base, judgment),
        python_backend=score_python_backend(profile),
        cloud_fullstack=score_cloud_fullstack(profile),
        github=min(_WEIGHTS.github_max, github_points),
        engineering_depth=score_engineering_depth(profile),
    )
