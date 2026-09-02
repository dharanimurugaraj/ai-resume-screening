"""Stage 2: resume bytes -> normalized raw text.

A malformed PDF must not crash the batch: callers should catch ParseError
per-file and record it in failed_resumes rather than letting it propagate.
"""

from __future__ import annotations

import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class ParseError(Exception):
    """Raised when a resume file cannot be turned into usable text."""


def parse_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
    except (PdfReadError, OSError, ValueError) as exc:
        raise ParseError(f"Could not open PDF: {exc}") from exc

    pages_text: list[str] = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception as exc:  # pypdf can raise assorted low-level errors
            raise ParseError(f"Could not extract text from a page: {exc}") from exc

    text = "\n".join(pages_text)
    text = _normalize_whitespace(text)

    if not text.strip():
        raise ParseError("No extractable text (resume may be a scanned image)")

    return text


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of 3+ blank lines, and trailing spaces on each line.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
