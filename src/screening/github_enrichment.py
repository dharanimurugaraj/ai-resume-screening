"""Stage 6: lightweight public GitHub enrichment.

GitHub is an additional positive signal only -- never a hard eligibility
requirement. Every failure mode (no username, 404, private/empty, rate
limit, network error) degrades to a GithubSummary with 0 points rather than
raising, so one bad/missing profile never affects the rest of the batch.

Caching is in-run only (a dict keyed by username): the spec only asks to
avoid repeated network calls within a single run, not persistent caching
across runs.
"""

from __future__ import annotations

import logging

import requests

from screening.config import settings
from screening.models import GithubSummary

logger = logging.getLogger(__name__)

_cfg = settings.github
_cache: dict[str, GithubSummary] = {}

_AI_RELEVANT_TERMS = (
    "python",
    "ai",
    "ml",
    "llm",
    "rag",
    "agent",
    "langchain",
    "langgraph",
    "nlp",
    "vector",
    "embedding",
    "torch",
    "tensorflow",
    "genai",
)


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    return headers


def enrich_github(username: str | None) -> GithubSummary:
    if not username:
        return GithubSummary(status="no_profile_found", summary="No GitHub profile found on resume.")

    if username in _cache:
        return _cache[username]

    summary = _fetch(username)
    _cache[username] = summary
    return summary


def _fetch(username: str) -> GithubSummary:
    try:
        user_resp = requests.get(
            f"{_cfg.api_base_url}/users/{username}",
            headers=_headers(),
            timeout=_cfg.request_timeout_seconds,
        )
    except requests.RequestException as exc:
        logger.warning("GitHub user lookup failed for %s: %s", username, exc)
        return GithubSummary(
            status="api_error", username=username, reason=str(exc), summary="GitHub API request failed."
        )

    if user_resp.status_code == 404:
        return GithubSummary(
            status="no_profile_found", username=username, summary="GitHub profile not found."
        )
    if user_resp.status_code == 403:
        return GithubSummary(
            status="rate_limited",
            username=username,
            reason="GitHub API rate limit exceeded",
            summary="GitHub enrichment skipped: rate limited.",
        )
    if user_resp.status_code != 200:
        return GithubSummary(
            status="api_error",
            username=username,
            reason=f"HTTP {user_resp.status_code}",
            summary="GitHub API returned an unexpected error.",
        )

    try:
        repos_resp = requests.get(
            f"{_cfg.api_base_url}/users/{username}/repos",
            params={"sort": "pushed", "per_page": 10},
            headers=_headers(),
            timeout=_cfg.request_timeout_seconds,
        )
        repos = repos_resp.json() if repos_resp.status_code == 200 else []
    except requests.RequestException as exc:
        logger.warning("GitHub repo lookup failed for %s: %s", username, exc)
        repos = []

    if not repos:
        return GithubSummary(
            status="private_or_empty",
            username=username,
            summary="GitHub profile has no visible public repositories.",
        )

    activity_score = _score_recent_activity(repos)
    repos_score = _score_maintained_repos(repos)

    return GithubSummary(
        status="ok",
        username=username,
        recent_activity_score=activity_score,
        maintained_repos_score=repos_score,
        summary=_summarize(activity_score, repos_score, repos),
    )


def _score_recent_activity(repos: list[dict]) -> int:
    import datetime

    most_recent = max((r.get("pushed_at") for r in repos if r.get("pushed_at")), default=None)
    if not most_recent:
        return 0
    pushed = datetime.datetime.fromisoformat(most_recent.replace("Z", "+00:00"))
    days = (datetime.datetime.now(datetime.timezone.utc) - pushed).days
    if days <= 30:
        return 5
    if days <= 90:
        return 4
    if days <= 180:
        return 3
    if days <= 365:
        return 2
    return 1


def _score_maintained_repos(repos: list[dict]) -> int:
    relevant = 0
    for repo in repos:
        if repo.get("fork"):
            continue
        haystack = " ".join(
            filter(
                None,
                [
                    (repo.get("language") or "").lower(),
                    (repo.get("description") or "").lower(),
                    " ".join(repo.get("topics") or []).lower(),
                ],
            )
        )
        if any(term in haystack for term in _AI_RELEVANT_TERMS):
            relevant += 1
    if relevant == 0:
        return 0
    if relevant <= 1:
        return 2
    if relevant <= 3:
        return 4
    return 5


def _summarize(activity_score: int, repos_score: int, repos: list[dict]) -> str:
    non_fork = [r for r in repos if not r.get("fork")]
    activity_desc = "Recently active" if activity_score >= 4 else (
        "Some recent activity" if activity_score >= 2 else "Little recent activity"
    )
    repo_desc = f"{len(non_fork)} maintained repositories reviewed"
    return f"{activity_desc}; {repo_desc}."
