from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.common import (
    KINDS, REPO_ROOT, SLICES, PCDEvalModel, TextEvalModel, eval_device,
    fc_correct, fc_margins_pcd, fc_margins_text, load_eval_qa, scramble_perm,
)


def run_forced_choice(kind: str, run: str, variant: str, batch_size: int,
                      seed: int, cells: list[str] | None = None) -> dict:
    device = eval_device()
    if kind == "pcd":
        m = PCDEvalModel(run, device)

        def margin_fn(qa, perm):
            return fc_margins_pcd(m, qa, variant, perm, batch_size)

        n_rows = len(m.z)
    else:
        m = TextEvalModel(run, device, reasoning_path=KINDS[kind].get("reasoning"))

        def margin_fn(qa, perm):
            return fc_margins_text(m, qa, variant, perm, batch_size)

        n_rows = len(m.ctx_cache)
    perm = scramble_perm(n_rows, seed) if variant == "scramble" else None

    qa = load_eval_qa(m.cfg)
    results: dict = {}
    for sf in SLICES:
        sub = qa[qa.split_fine == sf]
        for qc in sorted(sub.question_class.unique()):
            if cells and f"{sf}:{qc}" not in cells:
                continue
            cell = sub[sub.question_class == qc]
            correct = fc_correct(margin_fn(cell, perm), cell["delta"])
            results.setdefault(sf, {})[qc] = {
                "n": int(len(cell)),
                "fc_sign_acc": round(float(correct.mean()), 4),
            }
            print(f"[fc] {kind}/{variant} {sf} {qc} "
                  f"n={len(cell)} acc={correct.mean():.4f}", flush=True)
    return {"kind": kind, "run": run, "variant": variant, "results": results}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", required=True, choices=["pcd", "f1", "f1prime", "f2"])
    p.add_argument("--variant", default="real",
                   choices=["real", "zeros", "scramble", "qonly"])
    p.add_argument("--run", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cells", nargs="+", default=None,
                   help="restrict to split_fine:question_class cells")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.variant in ("zeros",) and args.kind != "pcd":
        raise SystemExit("zeros variant applies to --kind pcd only")
    if args.variant == "qonly" and args.kind == "pcd":
        raise SystemExit("qonly variant applies to text kinds only")

    run = args.run or KINDS[args.kind]["run"]
    out = run_forced_choice(args.kind, run, args.variant, args.batch_size,
                            args.seed, args.cells)
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / f"artifacts/baselines/evals/fc_{args.kind}_{args.variant}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[fc] -> {out_path}")


if __name__ == "__main__":
    main()
