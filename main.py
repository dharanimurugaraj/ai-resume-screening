"""CLI entrypoint.

Usage:
    python main.py --input ./resumes --output ./output/results.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from screening.pipeline import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Resume Screening & Ranking System")
    parser.add_argument("--input", required=True, help="Directory containing resume PDFs")
    parser.add_argument("--output", required=True, help="Path to write results.json")
    parser.add_argument(
        "--verbose", action="store_true", help="Enable INFO-level logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    run(args.input, args.output)
    print(f"Done. Results written to {args.output}")


if __name__ == "__main__":
    main()
