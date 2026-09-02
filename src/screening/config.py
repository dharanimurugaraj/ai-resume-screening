"""Central configuration: weights, thresholds, model names, env var names.

Business logic modules should import `settings` from here rather than reading
os.environ or hard-coding numbers directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScoreWeights:
    ai_project_depth_max: int = 40
    python_backend_max: int = 30
    cloud_fullstack_max: int = 15
    github_max: int = 10
    engineering_depth_max: int = 5

    # AI project depth split between deterministic base and LLM judgment
    ai_deterministic_max: int = 20
    ai_llm_max: int = 20


@dataclass(frozen=True)
class GithubScoring:
    recent_activity_max: int = 5
    maintained_repos_max: int = 5
    api_base_url: str = "https://api.github.com"
    request_timeout_seconds: float = 6.0


@dataclass(frozen=True)
class Settings:
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    github: GithubScoring = field(default_factory=GithubScoring)

    # LLM provider config (Gemini). Adapter reads these, never hard-codes.
    gemini_api_key: str | None = field(
        default_factory=lambda: os.environ.get("GEMINI_API_KEY")
    )
    gemini_model: str = field(
        default_factory=lambda: os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    )
    llm_enabled: bool = field(
        default_factory=lambda: bool(os.environ.get("GEMINI_API_KEY"))
    )

    github_token: str | None = field(
        default_factory=lambda: os.environ.get("GITHUB_TOKEN")
    )


settings = Settings()
