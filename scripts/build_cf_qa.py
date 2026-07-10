#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cf_dataset.build import run  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--minimal-run", action="store_true")
    args = p.parse_args()
    run(args.config, minimal_run=args.minimal_run)


if __name__ == "__main__":
    main()
