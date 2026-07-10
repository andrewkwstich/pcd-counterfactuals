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
from src.subject.probe import run_probe  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--adapter", default=None,
                   help="LoRA adapter dir (omit to probe the base model)")
    p.add_argument("--out", default=None, help="report path (default: next to adapter)")
    p.add_argument("--minimal-run", action="store_true")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    pc = cfg.get("probe", {})
    n = pc.get("n_examples", 4096)
    batch_size = pc.get("batch_size", 32)
    if args.minimal_run:
        n, batch_size = 64, 8

    df = load_split(str(REPO_ROOT / cfg["data"]["path"]),
                    pc.get("split", cfg["data"]["val_split"]), n)
    model, tokenizer, _ = load_subject(cfg["model"], args.adapter)
    report = run_probe(
        model, tokenizer, df,
        read_layer=cfg["read_layer"],
        anchor_token_id=cfg["anchor"]["token_id"],
        batch_size=batch_size,
        test_frac=pc.get("test_frac", 0.2),
        threshold_r2=pc.get("threshold_r2", 0.8),
    )
    report["adapter"] = args.adapter
    report["minimal_run"] = args.minimal_run

    if args.out:
        out = Path(args.out)
    elif args.adapter:
        out = Path(args.adapter).resolve().parent / "localization_probe.json"
    else:
        out = REPO_ROOT / "artifacts/subject/localization_probe_base.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "probes"}, indent=2))
    print("probes:", json.dumps(report["probes"], indent=2))
    print(f"[probe] report -> {out}")
    if not args.minimal_run and not report["gate_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
