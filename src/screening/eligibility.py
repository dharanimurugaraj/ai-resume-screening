"""Stage 4: deterministic hard eligibility filter.

A candidate is eligible only when BOTH hold:
  A. Genuine Python evidence.
  B. At least one STRONG AI/agentic/RAG signal (generic terms like "AI",
     "prompt engineering", "OpenAI API", "HuggingFace", "Transformers" are
     tracked but never sufficient on their own -- they are easy to drop into
     a skills list without having built anything real).

Every keyword check uses word-boundary regex, never a bare substring test:
naive `"rag" in text.lower()` matches "leveraged", "average", "storage",
"Kharagpur" -- verified against this project's actual resume dataset (43 of
50 false-matched vs. 26 of 50 with word boundaries). Presence of JS/Java/
React/etc. is never inspected here, so it can never cause a rejection.
"""

from __future__ import annotations

import re

from screening.models import CandidateProfile, EligibilityResult

PYTHON_PATTERNS: list[str] = [
    r"\bpython\b",
    r"\bpythonic\b",
]

# Concrete, hard-to-fake evidence of a real AI/LLM/RAG/agentic system.
STRONG_AI_PATTERNS: list[str] = [
    r"\blangchain\b",
    r"\blanggraph\b",
    r"\bllama\s?index\b",
    r"\bgoogle\s+adk\b",
    r"\bagent\s+developer\s+kit\b",
    r"\bretrieval[\s-]augmented\s+generation\b",
    r"\brag\b",
    r"\brag[\s-]pipelines?\b",
    r"\bvector\s+(search|database|store|embeddings?)\b",
    r"\bsemantic\s+search\b",
    r"\bembeddings?\b",
    r"\btool[\s-]calling\b",
    r"\bfunction[\s-]calling\b",
    r"\bmulti[\s-]agent\b",
    r"\bautonomous\s+agents?\b",
    r"\bagentic\b",
    r"\bai\s+agents?\b",
    r"\bllm\s+evaluation\b",
    r"\bevaluation\s+pipelines?\b",
    r"\bpgvector\b",
    r"\bpinecone\b",
    r"\bchroma\s?db\b",
    r"\bfaiss\b",
    r"\bweaviate\b",
    r"\bqdrant\b",
    r"\bmilvus\b",
]

# Signals that are real but too generic/easy-to-list to prove a meaningful
# project on their own. Recorded for scoring context, never for eligibility.
WEAK_AI_PATTERNS: list[str] = [
    r"\bai\b",
    r"\bartificial\s+intelligence\b",
    r"\bprompt\s+engineering\b",
    r"\bopenai(\s+api)?\b",
    r"\bhugging\s?face\b",
    r"\btransformers?\b",
    r"\bllms?\b",
    r"\blarge\s+language\s+models?\b",
    r"\bgpt\b",
    r"\bchatgpt\b",
    r"\bgen(erative)?[\s-]?ai\b",
    r"\bmachine\s+learning\b",
    r"\bdeep\s+learning\b",
    r"\bnlp\b",
]

# Additional skills tracked purely for matched_skills / scoring display.
DISPLAY_SKILL_PATTERNS: dict[str, str] = {
    "JavaScript": r"\bjavascript\b",
    "TypeScript": r"\btypescript\b",
    "Java": r"\bjava\b(?!script)",
    "React": r"\breact(\.js)?\b",
    "Next.js": r"\bnext\.js\b",
    "FastAPI": r"\bfastapi\b",
    "Flask": r"\bflask\b",
    "Django": r"\bdjango\b",
    "PostgreSQL": r"\bpostgres(ql)?\b",
    "MySQL": r"\bmysql\b",
    "MongoDB": r"\bmongodb\b",
    "Redis": r"\bredis\b",
    "Docker": r"\bdocker\b",
    "Kubernetes": r"\bkubernetes\b|\bk8s\b",
    "GCP": r"\bgcp\b|\bgoogle\s+cloud\b",
    "AWS": r"\baws\b",
    "Azure": r"\bazure\b",
}


def _find_matches(patterns: list[str], text: str) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        found = re.search(pattern, text, re.IGNORECASE)
        if found:
            matches.append(found.group(0))
    return matches


def check_eligibility(profile: CandidateProfile) -> EligibilityResult:
    text = profile.raw_text

    python_evidence = _find_matches(PYTHON_PATTERNS, text)
    strong_ai_evidence = _find_matches(STRONG_AI_PATTERNS, text)
    weak_ai_evidence = _find_matches(WEAK_AI_PATTERNS, text)

    has_python = bool(python_evidence)
    has_strong_ai = bool(strong_ai_evidence)

    matched_skills = _matched_display_skills(text)
    if has_python:
        matched_skills = ["Python"] + matched_skills
    matched_skills += [m for m in strong_ai_evidence if m.lower() not in [s.lower() for s in matched_skills]]

    rejection_reasons: list[str] = []
    if not has_python:
        rejection_reasons.append("No evidence of Python stack")
    if not has_strong_ai:
        rejection_reasons.append(
            "No meaningful AI/agentic project evidence "
            "(generic AI/LLM/prompt-engineering mentions alone are not sufficient)"
        )

    return EligibilityResult(
        eligible=has_python and has_strong_ai,
        matched_skills=matched_skills,
        rejection_reasons=rejection_reasons,
        python_evidence=python_evidence,
        ai_evidence_strong=strong_ai_evidence,
        ai_evidence_weak=weak_ai_evidence,
    )


def _matched_display_skills(text: str) -> list[str]:
    found: list[str] = []
    for label, pattern in DISPLAY_SKILL_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            found.append(label)
    return found
