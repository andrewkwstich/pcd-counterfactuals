from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.common import (
    KINDS, REPO_ROOT, PCDEvalModel, eval_device, gen_metrics, generate_pcd,
    load_eval_qa, class_sample, scramble_perm,
)

NOTE = "sense1_test nonzero-delta only; greedy decode; scramble = random other applicant z"


def _variant_metrics(preds, deltas) -> dict:
    m = gen_metrics(preds, deltas)
    return {
        "parse_rate": round(m["parse_rate"], 3),
        "pred_zero_rate": round(m["pred_zero_rate"], 3),
        "sign_acc_incl_zero_preds": round(m["sign_acc"], 3),
        "sign_acc_nonzero_preds": round(m["sign_acc_committed"], 3),
        "mae": round(m["mae"], 1),
        "predict_zero_baseline_mae": round(m["zero_baseline_mae"], 1),
        "n": m["n"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=KINDS["pcd"]["run"])
    p.add_argument("--per-class", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    m = PCDEvalModel(args.run, eval_device())
    qa = load_eval_qa(m.cfg)
    sample = class_sample(qa, args.per_class, args.seed)
    perm = scramble_perm(len(m.z), args.seed)
    deltas = sample["delta"].to_numpy()

    results = {}
    for variant, pm in [("real", None), ("scramble", perm)]:
        preds = generate_pcd(m, sample, variant, pm, args.batch_size,
                             args.max_new_tokens)
        results[variant] = _variant_metrics(preds, deltas)
        print(f"[sanity] {variant} {results[variant]}", flush=True)

    out = {"results": results, "note": NOTE}
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "artifacts/baselines/evals/eval_e3_gen_results.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[sanity] -> {out_path}")


if __name__ == "__main__":
    main()
