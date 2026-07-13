from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.common import (
    GEN_SLICES, KINDS, REPO_ROOT, PCDEvalModel, TextEvalModel, eval_device,
    gen_metrics, generate_pcd, generate_text, load_eval_qa, sample_cell,
)


def run_generation(kind: str, run: str, n_per_cell: int, batch_size: int,
                   max_new_tokens: int, seed: int) -> dict:
    device = eval_device()
    spec = KINDS[kind]
    if kind == "pcd":
        m = PCDEvalModel(run, device)

        def gen_fn(qa):
            return generate_pcd(m, qa, "real", None, batch_size, max_new_tokens)
    else:
        m = TextEvalModel(run, device, reasoning_path=spec.get("reasoning"))

        def gen_fn(qa):
            return generate_text(m, qa, "real", None, batch_size, max_new_tokens)
    zero_baseline = spec.get("zero_baseline", True)

    qa = load_eval_qa(m.cfg)
    results: dict = {}
    for sf in GEN_SLICES:
        sub = qa[qa.split_fine == sf]
        for qc in sorted(sub.question_class.unique()):
            cell = sample_cell(sub[sub.question_class == qc], n_per_cell, seed)
            metrics = gen_metrics(gen_fn(cell), cell["delta"])
            rec = {
                "n": metrics["n"],
                "sign_acc": round(metrics["sign_acc"], 3),
                "sign_acc_committed": round(metrics["sign_acc_committed"], 3),
                "pred_zero_rate": round(metrics["pred_zero_rate"], 3),
                "mae": round(metrics["mae"], 0),
            }
            if zero_baseline:
                rec["zero_baseline_mae"] = round(metrics["zero_baseline_mae"], 0)
            results.setdefault(sf, {})[qc] = rec
            print(f"[gen] {kind} {sf} {qc} {rec}", flush=True)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", required=True, choices=sorted(KINDS))
    p.add_argument("--run", default=None)
    p.add_argument("--n-per-cell", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    run = args.run or KINDS[args.kind]["run"]
    results = run_generation(args.kind, run, args.n_per_cell, args.batch_size,
                             args.max_new_tokens, args.seed)
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / f"artifacts/baselines/evals/cmp_{args.kind}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[gen] -> {out_path}")


if __name__ == "__main__":
    main()
