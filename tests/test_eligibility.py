from screening.eligibility import check_eligibility
from screening.models import CandidateProfile


def _profile(text: str) -> CandidateProfile:
    return CandidateProfile(raw_text=text, skills_text=text, projects_text=text)


def test_python_only_no_ai_is_rejected():
    text = "Skills: Python, Django, PostgreSQL. Built a to-do list web app."
    result = check_eligibility(_profile(text))
    assert result.eligible is False
    assert "No meaningful AI/agentic project evidence" in " ".join(
        result.rejection_reasons
    )
    assert not any("Python" in r for r in result.rejection_reasons)


def test_ai_only_no_python_is_rejected():
    text = (
        "Skills: JavaScript, React, Node.js. Built a LangChain RAG pipeline "
        "with vector search over documents using LlamaIndex."
    )
    result = check_eligibility(_profile(text))
    assert result.eligible is False
    assert "No evidence of Python stack" in result.rejection_reasons


def test_python_and_strong_ai_is_eligible():
    text = (
        "Skills: Python, FastAPI. Built a LangGraph multi-agent workflow "
        "with tool-calling and a RAG pipeline backed by pgvector."
    )
    result = check_eligibility(_profile(text))
    assert result.eligible is True
    assert result.rejection_reasons == []


def test_python_and_ai_eligible_even_with_js_java_react_present():
    text = (
        "Skills: Python, Java, JavaScript, React, Next.js, TypeScript. "
        "Implemented an agentic RAG system with LangChain and vector embeddings, "
        "then built the frontend in React and Next.js."
    )
    result = check_eligibility(_profile(text))
    assert result.eligible is True
    assert "Java" in result.matched_skills
    assert "React" in result.matched_skills


def test_weak_generic_ai_terms_alone_are_not_sufficient():
    text = (
        "Skills: Python, AI, Prompt Engineering, OpenAI API, HuggingFace, Transformers. "
        "Used ChatGPT to help write code faster."
    )
    result = check_eligibility(_profile(text))
    assert result.eligible is False
    assert "No meaningful AI/agentic project evidence" in " ".join(
        result.rejection_reasons
    )
    # they should still be tracked as weak evidence for scoring context
    assert result.ai_evidence_weak


def test_rag_false_positive_words_do_not_count_as_ai_evidence():
    # "leveraged"/"average"/"storage" all contain the substring "rag" but
    # have nothing to do with Retrieval-Augmented Generation.
    text = (
        "Skills: Python. Leveraged cloud storage to cut average latency by 30%."
    )
    result = check_eligibility(_profile(text))
    assert result.eligible is False
    assert result.ai_evidence_strong == []


def test_standalone_rag_acronym_counts_as_strong_evidence():
    text = "Skills: Python. Built a RAG system for document Q&A."
    result = check_eligibility(_profile(text))
    assert result.eligible is True
    assert any("rag" in m.lower() for m in result.ai_evidence_strong)


def test_java_and_javascript_are_distinguished_in_matched_skills():
    text = "Skills: Python, Java, JavaScript. Built an AI agent with LangChain."
    result = check_eligibility(_profile(text))
    assert "Java" in result.matched_skills
    assert "JavaScript" in result.matched_skills
