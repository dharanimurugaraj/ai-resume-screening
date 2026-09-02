"""Stage 1: discover resume files in the input directory.

Deliberately dumb: lists files, reads bytes, computes a content hash so the
pipeline can flag byte-identical duplicate submissions. No parsing here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from screening.models import ResumeFile

SUPPORTED_EXTENSIONS = {".pdf"}


def discover_resumes(input_dir: str | Path) -> list[ResumeFile]:
    """Return every supported resume file in input_dir, sorted by filename."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    resumes: list[ResumeFile] = []
    for path in sorted(input_path.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        resumes.append(
            ResumeFile(path=str(path), filename=path.name, content_hash=content_hash)
        )
    return resumes


def find_duplicate_files(resumes: list[ResumeFile]) -> dict[str, list[str]]:
    """Group filenames by content hash; return only hashes with >1 file."""
    by_hash: dict[str, list[str]] = {}
    for resume in resumes:
        by_hash.setdefault(resume.content_hash, []).append(resume.filename)
    return {h: names for h, names in by_hash.items() if len(names) > 1}
