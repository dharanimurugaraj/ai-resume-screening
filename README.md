# AI Resume Screening & Ranking System

Ingests a folder of resumes (PDF), filters candidates against a deterministic
Python + AI/agentic eligibility bar, scores eligible candidates on a
100-point rubric, enriches with public GitHub activity, and produces a
ranked `results.json`.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
cp .env.example .env          # then fill in GEMINI_API_KEY if you want LLM scoring
```

Python 3.11+ recommended (developed and tested on 3.14).

## Run

```bash
python main.py --input resume_dataset_50/resumes --output output/results.json
```

Add `--verbose` for progress/warning logs (duplicate resumes, GitHub/LLM
failures, etc.). The pipeline runs fully without any environment variables
set: with no `GEMINI_API_KEY`, AI-project-depth scoring falls back to a
deterministic-only score (see below) instead of failing.

## Run tests

```bash
python -m pytest
```

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | No | Enables the single structured LLM call per eligible candidate (AI project-depth judgment + summary). Without it, scoring is 100% deterministic. |
| `GEMINI_MODEL` | No | Defaults to `gemini-2.5-flash`. |
| `GITHUB_TOKEN` | No | Raises the GitHub API rate limit from 60/hr to 5000/hr. GitHub enrichment works fine without it for a single run over ~50 resumes. |

No credentials are ever hard-coded; see `.env.example`.

---

## Design Decisions

### Filtering strategy

Eligibility requires **both** genuine Python evidence and at least one
**strong** AI/agentic signal, checked with word-boundary regex over the full
resume text — never a bare substring test. This matters concretely: a naive
`"rag" in text.lower()` check false-matches "leveraged", "average",
"storage", "Kharagpur" — verified directly against this project's 50-resume
dataset (43/50 files false-matched vs. 26/50 with `\brag\b`). Every keyword
list in `eligibility.py` uses the same discipline.

AI evidence is split into **strong** signals (LangChain, LangGraph,
LlamaIndex, RAG, retrieval-augmented generation, vector search/embeddings,
tool-calling, multi-agent, agentic, evaluation pipelines, named vector
stores like pgvector/Pinecone/FAISS/Weaviate/Chroma/Qdrant/Milvus) and
**weak** signals (bare "AI", "prompt engineering", "OpenAI API",
"HuggingFace", "Transformers", "LLM", "machine learning", "deep learning",
etc.). Only strong signals count toward eligibility — a skills list that
only says "AI, Prompt Engineering, OpenAI API" is rejected, because none of
those prove a real agentic/RAG system was built. Weak signals are still
recorded and surfaced (`ai_evidence_weak`) for scoring context.

Because eligibility only checks for *presence* of Python + strong-AI terms
and never checks for the *absence* of other languages, JavaScript, Java,
React, and Next.js can never cause a rejection by construction — no special
"ignore these" logic is needed.

**Known limitation, by design, not a bug:** at least one resume in the
dataset shows deep RAG/pgvector/LLM-integration language but never mentions
"Python" anywhere (its stack is Node.js/TypeScript). Per the assignment's
explicit AND rule, this candidate is correctly rejected despite strong AI
evidence — a deliberate false negative, not an extraction failure.

### Scoring strategy

Deterministic wherever the rubric allows it to be, with exactly **one**
optional LLM call per eligible candidate for the genuinely judgment-based
half of AI-project-depth scoring (see below).

- **Python & Backend (30)**, **Cloud/Deployment (15)**, **Engineering Depth
  (5)** are all keyword/context-weighted: each term is checked in
  `projects_text` first (full weight) and only at reduced weight if it
  appears solely in the flat skills list — directly implementing "reward
  evidence in projects/internships over keyword-only skill lists."
- **AI/Agentic/RAG Depth (40)** is split 20/20: a deterministic base (count
  of distinct strong AI terms, weighted toward project-text occurrences) plus
  an LLM judgment score. A `thin_wrapper`/`tutorial_style` classification is
  hard-capped at 5 LLM points regardless of what the model returns — a
  deterministic safety net so one overly generous model call can't inflate a
  shallow project's score.
- **GitHub (10)** comes from enrichment (below), capped and never
  eligibility-gating.
- All per-category caps are enforced with `min(cap, ...)`, so a total is
  always reproducible from its breakdown by simple addition — no hidden
  normalization or curve.

### LLM usage

Exactly one structured call per eligible candidate (`llm/judge.py`), never
per-resume for ineligible candidates (they're filtered out deterministically
first — cheaper and avoids the LLM having any say in eligibility). The call
returns one `AIJudgment` object — classification, points, evidence,
reasoning, `project_summary`, `strengths`, `concerns` — together, so no
separate call is spent just generating a summary.

The adapter (`llm/adapter.py`) exposes a single function,
`call_structured(prompt, schema, system) -> T | None`, and every other
module depends only on that signature, not on the Gemini SDK directly.
Swapping providers means editing `adapter.py` and the env var names in
`config.py`; nothing in `scoring.py` or `pipeline.py` changes.

**Fail-soft by construction:** if `GEMINI_API_KEY` is unset, the client
fails to initialize, the API call errors, or the response doesn't parse
against the schema, `call_structured` returns `None`. The pipeline then uses
a documented deterministic fallback: AI-depth score is the deterministic
base scaled to the full 40-point range, and `project_summary` is built from
the first project-text sentence containing a strong AI term. One candidate's
LLM failure is caught again at the pipeline loop boundary as a second safety
net and never stops the batch.

### GitHub scoring

`github.com/<username>` is extracted with a simple regex. Enrichment calls
the public GitHub REST API (`/users/{u}`, `/users/{u}/repos`), scored 0-5 for
recency of the most recently pushed repo and 0-5 for count of non-fork
repos whose language/description/topics look Python/AI-relevant. Every
failure mode (no profile found, 404, 403 rate-limited, private/empty,
network error) degrades to `GithubSummary(status=..., 0 points)` rather than
raising — GitHub is purely additive, never a gate, matching the spec. Caching
is in-run only (a dict keyed by username), since the spec only asks to avoid
repeated calls within one run.

---

## If I Had More Time

1. **Better name extraction for the ~2 remaining edge cases.** Two resumes in
   the dataset put the name in a position pypdf's text-extraction order
   doesn't surface near the top of the text (likely a sidebar/table layout);
   the current heuristic falls back to the first plausible-looking line,
   which is occasionally a summary fragment. A small LLM-assisted
   name/contact extraction pass (with the current regex as a fallback) would
   close this gap without weakening the deterministic path.
2. **Column-aware PDF parsing.** At least one resume uses a two-column
   layout that `pypdf`/`pdftotext` interleave into partially garbled text.
   Keyword-presence checks (eligibility) survive this fine, but section-aware
   extraction (skills vs. projects attribution) degrades. A layout-aware
   extractor (e.g. clustering text by x-position before reading order) would
   fix this properly.
3. **Persistent GitHub/LLM caching across runs**, keyed by resume content
   hash, so re-running the pipeline on an unchanged folder doesn't re-spend
   API quota.
4. **Bounded-concurrency batch processing** for the GitHub and LLM calls
   (currently sequential) — would meaningfully cut wall-clock time over ~50
   resumes without complicating the failure-isolation guarantees.
5. **A small FastAPI wrapper** (`POST /screen`, `GET /results`) over the
   existing `pipeline.run()`, since the core logic is already decoupled from
   the CLI entrypoint.

---

## Project structure

```
main.py                        CLI entrypoint
src/screening/
  config.py                    weights, thresholds, model name, env var names
  models.py                    Pydantic schemas shared across every stage
  ingestion.py                 discover resume files, hash for dup-detection
  parsing.py                   PDF bytes -> normalized text (fails soft)
  extraction.py                text -> name/email/github/skills/projects
  eligibility.py                deterministic Python + AI hard filter
  scoring.py                   100-point deterministic scoring rules
  github_enrichment.py         public GitHub API, 0-10 pts, fail-soft
  llm/
    adapter.py                 provider-agnostic call_structured()
    schemas.py                 AIJudgment schema + prompt
    judge.py                   the one LLM call site
  pipeline.py                  orchestrates every stage, per-file isolation
  output.py                    assembles + writes results.json
tests/                         eligibility, scoring, extraction, failure isolation
```

## Known limitations

- PDF only (per assignment scope); DOCX/TXT not implemented.
- No OCR — a scanned/image-only PDF would fail parsing gracefully (recorded
  in `failed_resumes`) rather than being read.
- Name extraction is regex-heuristic; see "If I Had More Time" above.
- No ground-truth labels exist for this dataset, so scoring/eligibility
  correctness is judged by internal consistency and the evidence strings in
  the output, not by comparison to an answer key.
