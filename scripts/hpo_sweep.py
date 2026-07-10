#!/usr/bin/env python
import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LRS = [5e-5, 1e-4, 2e-4, 4e-4]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--lrs", nargs="+", type=float, default=DEFAULT_LRS)
    p.add_argument("--pilot-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--tie-tol", type=float, default=0.02,
                   help="relative MAE tolerance within which ties go to lower LR")
    p.add_argument("--minimal-run", action="store_true",
                   help="2 LRs x 10 steps to validate the sweep path")
    args = p.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    lrs = sorted(args.lrs)
    if args.minimal_run:
        lrs = lrs[:2]

    sweep_root = REPO_ROOT / "artifacts/subject/hpo"
    sweep_root.mkdir(parents=True, exist_ok=True)

    results = []
    for lr in lrs:
        cfg = copy.deepcopy(base_cfg)
        name = f"hpo-lr{lr:g}"
        cfg["run"] = {**cfg.get("run", {}), "name": name,
                      "output_root": "artifacts/subject/hpo"}
        t = cfg["train"]
        t["learning_rate"] = lr
        t["max_steps"] = args.pilot_steps
        t["eval_steps"] = args.eval_steps
        t["save_strategy"] = "no"
        cfg["data"]["max_val_examples"] = min(
            cfg["data"].get("max_val_examples") or 1024, 1024)
        cfg_path = sweep_root / f"{name}.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)

        cmd = [sys.executable, str(REPO_ROOT / "scripts/train_subject.py"),
               "--config", str(cfg_path)]
        if args.minimal_run:
            cmd.append("--minimal-run")
        print(f"\n[hpo] ===== lr={lr:g} -> {name} =====", flush=True)
        env = {**os.environ, "WANDB_RUN_GROUP": "subject-hpo"}
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
        run_dir = sweep_root / (name + ("-minimal" if args.minimal_run else ""))
        rec = {"lr": lr, "run_dir": str(run_dir), "returncode": proc.returncode}
        fm = run_dir / "final_metrics.json"
        if proc.returncode == 0 and fm.exists():
            final = json.loads(fm.read_text())
            rec["eval_loss"] = final.get("eval_loss")
            rec.update({f"decode_{k}": v for k, v in final.get("decode", {}).items()})
        else:
            rec["error"] = "run failed or final_metrics.json missing"
        results.append(rec)

    ok = [r for r in results
          if r.get("decode_mae") is not None and (r.get("decode_parse_rate") or 0) >= 0.999]
    winner = None
    if ok:
        best_mae = min(r["decode_mae"] for r in ok)
        contenders = [r for r in ok if r["decode_mae"] <= best_mae * (1 + args.tie_tol)]
        winner = min(contenders, key=lambda r: r["lr"])

    summary = {
        "proposal": "short-budget LR ladder",
        "pilot_steps": args.pilot_steps,
        "tie_tol": args.tie_tol,
        "results": results,
        "winner_lr": winner["lr"] if winner else None,
        "winner_run_dir": winner["run_dir"] if winner else None,
        "note": "run full training with train.learning_rate = winner_lr",
    }
    out = sweep_root / ("hpo_summary-minimal.json" if args.minimal_run else "hpo_summary.json")
    out.write_text(json.dumps(summary, indent=2))

    print("\n[hpo] ===== summary =====")
    for r in results:
        print(f"  lr={r['lr']:g}: mae={r.get('decode_mae')}, medae={r.get('decode_medae')}, "
              f"r2={r.get('decode_r2')}, parse={r.get('decode_parse_rate')}, "
              f"eval_loss={r.get('eval_loss')}, rc={r['returncode']}")
    print(f"[hpo] winner: lr={summary['winner_lr']}  ({out})")
    if winner is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
