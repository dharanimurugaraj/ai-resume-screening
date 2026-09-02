"""Stage orchestration: ties ingestion -> ... -> output together.

The one hard rule: an exception while processing resume N must never stop
resumes N+1..len(resumes) from being processed. Every per-file failure is
caught at this loop boundary and recorded in `failed_resumes`.
"""

from __future__ import annotations

import logging
import re

from screening.eligibility import check_eligibility
from screening.extraction import extract_profile
from screening.github_enrichment import enrich_github
from screening.ingestion import discover_resumes, find_duplicate_files
from screening.llm.judge import judge_ai_project
from screening.models import CandidateResult, FailedResume
from screening.output import build_report, write_report
from screening.parsing import ParseError, parse_pdf
from screening.scoring import build_score_breakdown

logger = logging.getLogger(__name__)


def run(input_dir: str, output_path: str) -> None:
    resumes = discover_resumes(input_dir)

    duplicate_groups = find_duplicate_files(resumes)
    for content_hash, filenames in duplicate_groups.items():
        logger.warning("Duplicate resume content across files: %s", filenames)

    candidate_results: list[CandidateResult] = []
    failed_resumes: list[FailedResume] = []
    successfully_parsed = 0

    for resume in resumes:
        try:
            result = _process_one(resume.path, resume.filename)
        except Exception as exc:  # last-resort safety net for this candidate
            logger.error("Unexpected failure processing %s", resume.filename, exc_info=True)
            failed_resumes.append(
                FailedResume(source_file=resume.filename, stage="pipeline", error=str(exc))
            )
            continue

        if isinstance(result, FailedResume):
            failed_resumes.append(result)
            continue

        successfully_parsed += 1
        candidate_results.append(result)

    report = build_report(
        total_resumes=len(resumes),
        successfully_parsed=successfully_parsed,
        candidate_results=candidate_results,
        failed_resumes=failed_resumes,
    )
    write_report(report, output_path)


def _process_one(path: str, filename: str) -> CandidateResult | FailedResume:
    try:
        text = parse_pdf(path)
    except ParseError as exc:
        return FailedResume(source_file=filename, stage="parsing", error=str(exc))

    try:
        profile = extract_profile(text)
    except Exception as exc:
        return FailedResume(source_file=filename, stage="extraction", error=str(exc))

    eligibility = check_eligibility(profile)

    if not eligibility.eligible:
        return CandidateResult(
            candidate_name=profile.name,
            source_file=filename,
            email=profile.email,
            eligible=False,
            matched_skills=eligibility.matched_skills,
            rejection_reasons=eligibility.rejection_reasons,
        )

    try:
        judgment = judge_ai_project(profile)
    except Exception:
        logger.warning("LLM judgment raised unexpectedly for %s", filename, exc_info=True)
        judgment = None

    github_summary = enrich_github(profile.github_username)
    breakdown = build_score_breakdown(profile, github_summary.total_points, judgment)

    if judgment is not None:
        project_summary = judgment.project_summary
        strengths = judgment.strengths
        concerns = judgment.concerns
    else:
        project_summary = _fallback_project_summary(profile)
        strengths = list(eligibility.ai_evidence_strong[:3])
        concerns = ["AI judgment unavailable: LLM call skipped or failed; scored from deterministic evidence only."]

    return CandidateResult(
        candidate_name=profile.name,
        source_file=filename,
        email=profile.email,
        eligible=True,
        matched_skills=eligibility.matched_skills,
        total_score=breakdown.total,
        score_breakdown=breakdown,
        project_summary=project_summary,
        github_summary=github_summary,
        strengths=strengths,
        concerns=concerns,
    )


def _fallback_project_summary(profile) -> str:
    for sentence in re.split(r"(?<=[.!?])\s+", profile.projects_text):
        sentence = sentence.strip()
        if len(sentence) > 20 and re.search(
            r"\b(rag|langchain|langgraph|agent|vector|embedding|llm)\b", sentence, re.IGNORECASE
        ):
            return sentence[:280]
    snippet = profile.projects_text.strip().replace("\n", " ")
    return snippet[:200] if snippet else "No project summary available."
