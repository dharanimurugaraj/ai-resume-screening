"""Structured-output schema(s) for LLM calls.

Only one schema is needed: a single call per eligible candidate returns the
AI project-depth judgment, evidence/reasoning, project summary, and
strengths/concerns together (see AIJudgment in screening.models).
"""

from __future__ import annotations

from screening.models import AIJudgment

AI_JUDGMENT_SYSTEM_PROMPT = """You are an experienced technical recruiter for an SDE Internship \
requiring strong Python fundamentals and practical AI/agentic-systems experience. \
You will be given the extracted text of one candidate resume that has already passed a \
deterministic Python + AI eligibility filter. Judge ONLY the depth and authenticity of their AI/LLM/RAG/agentic work.

Classify their strongest AI project into exactly one category:
- "thin_wrapper": little more than a single call to an LLM/API with no real workflow, retrieval, \
state, tool use, evaluation, or product logic.
- "tutorial_style": follows a well-known tutorial/course pattern with no evidence of original \
extension, ownership, or non-trivial customization.
- "applied_project": a real, working AI feature or project with some retrieval/tools/state/evaluation, \
but modest scope.
- "production_grade_system": a system with meaningful orchestration -- e.g. agents, RAG with real \
retrieval, tool-calling, multi-step/multi-agent state, evaluation, or production concerns.

Assign `points` from 0-20 for AI project depth (thin_wrapper/tutorial_style should typically score \
0-5; applied_project 6-13; production_grade_system 14-20). Do not award high points for a framework \
name merely appearing in a skills list -- require evidence of how it was actually used.

`evidence` must quote or closely paraphrase the specific resume text that justifies the classification. \
`reasoning` is 1-2 sentences explaining the score. `project_summary` is one concise sentence describing \
their strongest AI project. `strengths` and `concerns` are short bullet phrases (0-4 each) covering the \
candidate's overall technical profile, not just the AI project."""


def build_ai_judgment_prompt(candidate_text: str) -> str:
    # Resumes run long; truncate defensively so a single call stays cheap
    # and within context limits even for multi-page resumes.
    truncated = candidate_text[:8000]
    return (
        "Resume text:\n---\n"
        f"{truncated}\n---\n\n"
        "Return your judgment as the structured AIJudgment schema."
    )


__all__ = ["AIJudgment", "AI_JUDGMENT_SYSTEM_PROMPT", "build_ai_judgment_prompt"]
