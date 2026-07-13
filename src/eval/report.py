from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.common import (
    CLASSES, KINDS, REPO_ROOT, PCDEvalModel, TextEvalModel, eval_device,
    gen_metrics, generate_pcd, generate_text, load_eval_qa, class_sample,
    scramble_perm,
)


def _variant_metrics(preds, deltas) -> dict:
    m = gen_metrics(preds, deltas)
    return {
        "sign_acc": round(m["sign_acc"], 4),
        "sign_acc_committed": round(m["sign_acc_committed"], 4),
        "pred_zero_rate": round(m["pred_zero_rate"], 4),
        "mae": round(m["mae"], 1),
        "n": m["n"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", required=True, choices=["pcd", "f1", "f1prime", "f2"])
    p.add_argument("--run", default=None)
    p.add_argument("--per-class", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    run = args.run or KINDS[args.kind]["run"]
    device = eval_device()
    if args.kind == "pcd":
        m = PCDEvalModel(run, device)

        def gen_fn(qa, variant, perm):
            return generate_pcd(m, qa, variant, perm, args.batch_size,
                                args.max_new_tokens)

        n_rows = len(m.z)
        report_name = "finetune_report.json"
    else:
        m = TextEvalModel(run, device)

        def gen_fn(qa, variant, perm):
            return generate_text(m, qa, variant, perm, args.batch_size,
                                 args.max_new_tokens)

        n_rows = len(m.ctx_cache)
        report_name = "baseline_report.json"

    qa = load_eval_qa(m.cfg)
    sample = class_sample(qa, args.per_class, args.seed)
    perm = scramble_perm(n_rows, args.seed)
    deltas = sample["delta"].to_numpy()

    real_preds = gen_fn(sample, "real", None)
    scramble_preds = gen_fn(sample, "scramble", perm)

    correct = np.array([
        p is not None and ((p > 0) == (d > 0)) and p != 0
        for p, d in zip(real_preds, deltas)
    ])
    by_class = {}
    classes = sample["question_class"].to_numpy()
    for c in CLASSES:
        mask = classes == c
        by_class[c] = round(float(correct[mask].mean()), 4)

    out = {
        "real": _variant_metrics(real_preds, deltas),
        "scramble": _variant_metrics(scramble_preds, deltas),
        "by_class_sign_acc_real": by_class,
    }
    if args.kind != "pcd":
        out["predict_zero_baseline_mae"] = round(float(np.abs(deltas).mean()), 1)
    train_report = json.loads((REPO_ROOT / run / report_name).read_text())
    out["val_qa_loss"] = train_report["val_qa_loss"]
    out["val_exact_match"] = train_report["val_exact_match"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[report] -> {out_path}")


if __name__ == "__main__":
    main()
