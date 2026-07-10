#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.subject.train import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to configs/subject_*.yaml")
    parser.add_argument("--minimal-run", action="store_true",
                        help="~10 steps on ~100 examples to validate the path")
    args = parser.parse_args()
    run(args.config, minimal_run=args.minimal_run)


if __name__ == "__main__":
    main()
