"""Stage 8: assemble the final ScreeningReport and write results.json."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from screening.models import CandidateResult, FailedResume, ScreeningReport
from screening.models import BatchSummary


def build_report(
    total_resumes: int,
    successfully_parsed: int,
    candidate_results: list[CandidateResult],
    failed_resumes: list[FailedResume],
) -> ScreeningReport:
    eligible = sorted(
        (c for c in candidate_results if c.eligible),
        key=lambda c: c.total_score or 0,
        reverse=True,
    )
    for rank, candidate in enumerate(eligible, start=1):
        candidate.rank = rank

    rejected = [c for c in candidate_results if not c.eligible]

    summary = BatchSummary(
        total_resumes=total_resumes,
        successfully_parsed=successfully_parsed,
        eligible=len(eligible),
        rejected=len(rejected),
        failed_unreadable=len(failed_resumes),
        generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )

    return ScreeningReport(
        batch_summary=summary,
        eligible_candidates=eligible,
        rejected_candidates=rejected,
        failed_resumes=failed_resumes,
    )


def write_report(report: ScreeningReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
