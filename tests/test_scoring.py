from screening.models import AIJudgment, CandidateProfile
from screening.scoring import (
    build_score_breakdown,
    score_ai_deterministic_base,
    score_ai_project_depth,
    score_cloud_fullstack,
    score_engineering_depth,
    score_python_backend,
)


def _profile(projects: str = "", skills: str = "") -> CandidateProfile:
    return CandidateProfile(
        raw_text=projects + "\n" + skills, projects_text=projects, skills_text=skills
    )


def test_python_backend_scores_higher_when_evidence_is_in_projects():
    rich = _profile(projects="Built a Python FastAPI service with PostgreSQL and Redis caching.")
    thin = _profile(skills="Python, FastAPI, PostgreSQL, Redis")
    assert score_python_backend(rich) > score_python_backend(thin)


def test_python_backend_capped_at_max():
    profile = _profile(
        projects=(
            "Python FastAPI Flask Django async await PostgreSQL Redis backend, "
            "with more Python FastAPI Flask Django everywhere."
        )
    )
    assert score_python_backend(profile) <= 30


def test_cloud_fullstack_rewards_docker_and_cloud_and_deployment():
    profile = _profile(projects="Deployed on GCP using Docker and Kubernetes with CI/CD.")
    score = score_cloud_fullstack(profile)
    assert 0 < score <= 15


def test_engineering_depth_capped_at_five():
    profile = _profile(
        projects=(
            "Added unit tests, caching, a queue with Celery, monitoring with "
            "Grafana, asyncio concurrency, and retry-based error handling."
        )
    )
    assert score_engineering_depth(profile) == 5


def test_ai_deterministic_base_rewards_distinct_strong_terms():
    weak = _profile(projects="Used AI and prompt engineering with OpenAI API.")
    strong = _profile(
        projects="Built a LangGraph multi-agent RAG pipeline with pgvector and tool-calling."
    )
    assert score_ai_deterministic_base(strong) > score_ai_deterministic_base(weak)
    assert score_ai_deterministic_base(strong) <= 20


def test_ai_project_depth_without_llm_scales_deterministic_base():
    assert score_ai_project_depth(10, None) == 20
    assert score_ai_project_depth(20, None) == 40


def test_ai_project_depth_thin_wrapper_penalized_even_if_llm_overscores():
    generous_but_wrong = AIJudgment(
        classification="thin_wrapper",
        points=18,
        evidence="calls an LLM API once",
        reasoning="thin wrapper",
        project_summary="Calls OpenAI API to summarize text.",
    )
    score = score_ai_project_depth(4, generous_but_wrong)
    assert score <= 4 + 5


def test_ai_project_depth_production_system_gets_full_credit():
    judgment = AIJudgment(
        classification="production_grade_system",
        points=20,
        evidence="stateful agentic workflow with retrieval and tool calling",
        reasoning="real system",
        project_summary="Built a stateful agentic workflow with retrieval and tool calling.",
        strengths=["Strong agentic project"],
        concerns=[],
    )
    score = score_ai_project_depth(20, judgment)
    assert score == 40


def test_build_score_breakdown_total_within_100():
    profile = _profile(
        projects=(
            "Python FastAPI PostgreSQL Redis async, deployed on GCP with Docker "
            "and CI/CD. Built a LangChain RAG agent with vector search."
        ),
        skills="Python, FastAPI, Docker, GCP, LangChain",
    )
    breakdown = build_score_breakdown(profile, github_points=10, judgment=None)
    assert breakdown.total <= 100
    assert breakdown.github == 10
