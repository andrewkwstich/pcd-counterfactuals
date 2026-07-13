from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.common import (
    KINDS, REPO_ROOT, PCDEvalModel, TextEvalModel, eval_device, fc_correct,
    fc_margins_pcd, fc_margins_text, label_sign_ceiling, load_eval_qa,
)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
    return {"ci95": [float(lo), float(hi)], "p_two_sided": min(p, 1.0)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pcd-run", default=KINDS["pcd"]["run"])
    p.add_argument("--baseline-run", default=KINDS["f1prime"]["run"])
    p.add_argument("--split", default="sense2_race")
    p.add_argument("--question-class", default="name_categorical")
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = eval_device()
    pcd = PCDEvalModel(args.pcd_run, device)
    qa = load_eval_qa(pcd.cfg)
    cell = qa[(qa.split_fine == args.split) & (qa.question_class == args.question_class)]
    print(f"[primary] cell {args.split} x {args.question_class} N={len(cell)}")

    pcd_correct = fc_correct(
        fc_margins_pcd(pcd, cell, "real", None, args.batch_size), cell["delta"]
    )
    del pcd
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    baseline = TextEvalModel(args.baseline_run, device)
    baseline_correct = fc_correct(
        fc_margins_text(baseline, cell, "real", None, args.batch_size), cell["delta"]
    )

    pcd_acc = float(pcd_correct.mean())
    baseline_acc = float(baseline_correct.mean())
    diff = pcd_acc - baseline_acc
    boot = paired_bootstrap(pcd_correct.astype(float), baseline_correct.astype(float),
                            args.resamples, args.seed)
    if boot["p_two_sided"] < 0.05:
        verdict = "PCD > ' (sig)" if diff > 0 else "' > PCD (sig)"
    else:
        verdict = "PCD ~ ' (ns)"

    out = {
        "cell": f"{args.split} x {args.question_class}",
        "N": int(len(cell)),
        "pcd_fc_sign_acc": round(pcd_acc, 4),
        "f1prime_fc_sign_acc": round(baseline_acc, 4),
        "diff_pcd_minus_f1prime": round(diff, 4),
        "ci95_diff": [round(boot["ci95"][0], 4), round(boot["ci95"][1], 4)],
        "p_two_sided": round(boot["p_two_sided"], 4),
        "label_sign_ceiling": round(label_sign_ceiling(cell), 3),
        "verdict": verdict,
    }
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "artifacts/baselines/evals/primary_test.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[primary] -> {out_path}")


if __name__ == "__main__":
    main()
