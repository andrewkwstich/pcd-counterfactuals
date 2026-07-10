#!/usr/bin/env python

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from src.subject.data import load_split  # noqa: E402
from src.subject.model_io import load_subject  # noqa: E402
from src.subject.validate import run_validation  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--adapter", default=None,
                   help="LoRA adapter dir (omit to validate the base model)")
    p.add_argument("--out", default=None, help="report path (default: next to adapter)")
    p.add_argument("--minimal-run", action="store_true")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    vc = cfg.get("validate", {})
    n = vc.get("n_examples")
    batch_size = vc.get("batch_size", 64)
    n_apps = vc.get("n_reliability_apps", 192)
    n_names = vc.get("n_reliability_names", 96)
    if args.minimal_run:
        n, batch_size, n_apps, n_names = 64, 16, 8, 8

    df = load_split(str(REPO_ROOT / cfg["data"]["path"]),
                    vc.get("split", cfg["data"]["val_split"]), n)
    model, tokenizer, _ = load_subject(cfg["model"], args.adapter)
    report = run_validation(
        model, tokenizer, df,
        batch_size=batch_size,
        n_reliability_apps=n_apps,
        n_reliability_names=n_names,
        thresholds=vc.get("thresholds"),
        max_new_tokens=cfg["train"].get("max_new_tokens", 8),
    )
    report["adapter"] = args.adapter
    report["minimal_run"] = args.minimal_run

    if args.out:
        out = Path(args.out)
    elif args.adapter:
        out = Path(args.adapter).resolve().parent / "validation_report.json"
    else:
        out = REPO_ROOT / "artifacts/subject/validation_report_base.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[validate] report -> {out}")
    if not args.minimal_run and not report["b3_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
