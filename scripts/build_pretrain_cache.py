#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pcd.data import build_cache  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="pcd_pretrain_*.yaml (for data.general source)")
    p.add_argument("--n-domain-docs", type=int, default=12000)
    p.add_argument("--n-general-docs", type=int, default=60000)
    p.add_argument("--max-stream", type=int, default=None, help="safety cap on docs streamed")
    p.add_argument("--out", default="data/pretrain_cache")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    build_cache(cfg, args.out, args.n_domain_docs, args.n_general_docs, args.max_stream)


if __name__ == "__main__":
    main()
