"""Stage 3 (deterministic part): raw text -> CandidateProfile.

Name/email/phone/GitHub extraction is regex-based and intentionally simple.
Section splitting (skills vs. projects) is best-effort: resumes have no
standard layout, so we fall back to using the whole text for both when we
can't confidently locate a section, rather than dropping information.
"""

from __future__ import annotations

import re

from screening.models import CandidateProfile

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?\d{10}\b")
GITHUB_RE = re.compile(
    r"github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]){0,38})", re.IGNORECASE
)

# Header contact-line noise: split the first line on these to isolate a name.
_HEADER_SPLIT_RE = re.compile(
    r"\s{2,}|(?=\+?\d{3,})|(?=[A-Za-z0-9._%+\-]+@)|\||(?=https?://)", re.UNICODE
)

_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")

# Non-name lines that some templates place before/around the name, or as a
# page artifact (bare page numbers). Extend cautiously.
_NAME_STOPWORDS = {
    "summary",
    "resume summary",
    "professional summary",
    "career aspiration",
    "career objective",
    "objective",
    "get in touch!",
    "get in touch",
    "email",
    "phone",
    "mobile",
    "linkedin",
    "github",
    "curriculum vitae",
    "resume",
}

_NAME_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z.'\-]*$")

# Contact-label tokens that sometimes glue directly onto the name with no
# separating space (e.g. "Vaibhav WakdeEmail:"). Stripped from the tail.
_TRAILING_LABELS = {"email", "linkedin", "github", "mobile", "phone", "contact"}

# Common English function words: a real name essentially never contains one
# of these as a standalone token, whereas a stray summary sentence usually
# does. Used to reject paragraph text that slipped past the earlier checks.
_SENTENCE_STOPWORDS = {
    "and",
    "the",
    "with",
    "for",
    "of",
    "is",
    "was",
    "in",
    "on",
    "to",
    "an",
    "a",
    "using",
    "like",
}

SECTION_HEADERS = {
    "skills": ["skills", "technical skills", "technologies", "tech stack"],
    "projects": [
        "projects",
        "project experience",
        "work experience",
        "professional experience",
        "experience",
        "internship",
        "internships",
    ],
}


def extract_profile(raw_text: str) -> CandidateProfile:
    email = _first_match(EMAIL_RE, raw_text)
    github_url, github_username = _extract_github(raw_text)
    name = _extract_name(raw_text)
    phone = _first_match(PHONE_RE, raw_text)

    skills_text = _extract_section(raw_text, SECTION_HEADERS["skills"])
    projects_text = _extract_section(raw_text, SECTION_HEADERS["projects"])

    return CandidateProfile(
        name=name,
        email=email,
        phone=phone,
        github_url=github_url,
        github_username=github_username,
        skills_text=skills_text or raw_text,
        projects_text=projects_text or raw_text,
        raw_text=raw_text,
    )


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(0) if match else None


def _extract_github(text: str) -> tuple[str | None, str | None]:
    match = GITHUB_RE.search(text)
    if not match:
        return None, None
    username = match.group(1)
    # Trailing path segments (repo names) sometimes attach without a
    # boundary character; usernames are capped at 39 chars by GitHub so this
    # mainly guards against picking up an org/repo slug as part of the name.
    return f"github.com/{username}", username


def _extract_name(text: str) -> str | None:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    for idx, line in enumerate(lines[:10]):
        if line.lower().strip(":!") in _NAME_STOPWORDS:
            continue
        if not re.search(r"[A-Za-z]{2,}", line):
            continue  # page numbers, bare punctuation, etc.

        # Some PDF templates emit one word per line (e.g. "ABHINAV" / "MISHRA"
        # on consecutive lines). Merge up to one following bare name-word;
        # a third consecutive short line is usually a city, not a surname.
        if _NAME_WORD_RE.match(line) and len(line) <= 20:
            words = [line]
            for follow in lines[idx + 1 : idx + 2]:
                if _NAME_WORD_RE.match(follow) and len(follow) <= 20:
                    words.append(follow)
            return " ".join(words)

        # Otherwise treat the line as a full header line and strip trailing
        # contact info: split camelCase runs first (PDFs sometimes merge
        # "Atul KumarDehradun" with no space at all), then split on the
        # usual contact-info separators and a leading location after a comma.
        candidate = _CAMEL_BOUNDARY_RE.sub(" ", line)
        candidate = _HEADER_SPLIT_RE.split(candidate)[0].strip(" -|:,")
        candidate = candidate.split(",")[0].strip()
        words = [w for w in candidate.split() if w.lower() not in _TRAILING_LABELS]
        candidate = " ".join(words)

        if not candidate or not re.search(r"[A-Za-z]{2,}", candidate):
            continue
        if len(words) > 6:
            continue  # reads as a sentence, not a name
        if any(len(w) > 1 and w.lower() in _SENTENCE_STOPWORDS for w in words):
            continue
        return candidate

    return None


def _extract_section(text: str, header_aliases: list[str]) -> str:
    """Return the text block starting at the first matching section header,
    up to the next recognized section header (any category), or end of text.
    """
    all_headers = [h for headers in SECTION_HEADERS.values() for h in headers]
    lines = text.splitlines()

    start_idx = None
    for idx, line in enumerate(lines):
        normalized = line.strip().lower().strip(":")
        if normalized in header_aliases:
            start_idx = idx + 1
            break

    if start_idx is None:
        return ""

    end_idx = len(lines)
    for idx in range(start_idx, len(lines)):
        normalized = lines[idx].strip().lower().strip(":")
        if normalized in all_headers:
            end_idx = idx
            break

    return "\n".join(lines[start_idx:end_idx]).strip()
