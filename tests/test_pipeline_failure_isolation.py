import json

import screening.pipeline as pipeline


def test_one_bad_resume_does_not_stop_the_batch(tmp_path, monkeypatch):
    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    good_file = resumes_dir / "candidate_good.pdf"
    bad_file = resumes_dir / "candidate_bad.pdf"
    rejected_file = resumes_dir / "candidate_reject.pdf"
    for f in (good_file, bad_file, rejected_file):
        f.write_bytes(b"%PDF-1.4 placeholder, content overridden by parse_pdf monkeypatch")

    fake_texts = {
        str(good_file): (
            "Jane Doe\njane@example.com\n"
            "Skills: Python, FastAPI, PostgreSQL\n"
            "Projects: Built a LangGraph multi-agent RAG pipeline with tool-calling and pgvector."
        ),
        str(rejected_file): "John Smith\njohn@example.com\nSkills: JavaScript, React, Node.js",
    }

    def fake_parse_pdf(path):
        if path == str(bad_file):
            from screening.parsing import ParseError

            raise ParseError("simulated corrupt PDF")
        return fake_texts[path]

    monkeypatch.setattr(pipeline, "parse_pdf", fake_parse_pdf)
    monkeypatch.setattr(pipeline, "judge_ai_project", lambda profile: None)
    monkeypatch.setattr(
        pipeline,
        "enrich_github",
        lambda username: __import__("screening.models", fromlist=["GithubSummary"]).GithubSummary(
            status="no_profile_found"
        ),
    )

    output_path = tmp_path / "results.json"
    pipeline.run(str(resumes_dir), str(output_path))

    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["batch_summary"]["total_resumes"] == 3
    assert report["batch_summary"]["failed_unreadable"] == 1
    assert report["batch_summary"]["eligible"] == 1
    assert report["batch_summary"]["rejected"] == 1

    assert report["failed_resumes"][0]["source_file"] == "candidate_bad.pdf"
    assert report["failed_resumes"][0]["stage"] == "parsing"

    assert report["eligible_candidates"][0]["candidate_name"] == "Jane Doe"
    assert report["rejected_candidates"][0]["candidate_name"] == "John Smith"


def test_llm_failure_does_not_crash_scoring(tmp_path, monkeypatch):
    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    good_file = resumes_dir / "candidate.pdf"
    good_file.write_bytes(b"placeholder")

    text = (
        "Jane Doe\njane@example.com\n"
        "Skills: Python, FastAPI\n"
        "Projects: Built a RAG agent with LangChain."
    )

    def raising_parse_pdf(path):
        return text

    def raising_judge(profile):
        raise RuntimeError("simulated LLM outage")

    monkeypatch.setattr(pipeline, "parse_pdf", raising_parse_pdf)
    monkeypatch.setattr(pipeline, "judge_ai_project", raising_judge)
    monkeypatch.setattr(
        pipeline,
        "enrich_github",
        lambda username: __import__("screening.models", fromlist=["GithubSummary"]).GithubSummary(
            status="no_profile_found"
        ),
    )

    output_path = tmp_path / "results.json"
    pipeline.run(str(resumes_dir), str(output_path))

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["batch_summary"]["eligible"] == 1
    assert report["eligible_candidates"][0]["total_score"] is not None
