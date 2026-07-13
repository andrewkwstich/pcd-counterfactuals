from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import torch.nn.functional as F

from src.eval.common import (
    KINDS, REPO_ROOT, PCDEvalModel, class_sample, eval_device, gen_metrics,
    generate_pcd, load_eval_qa, load_qa, scramble_perm,
)
from src.pcd.finetune import collate_qa

NOTE = "trained at k=16; k varied at encode time only (paper)"


@torch.no_grad()
def teacher_forced(m: PCDEvalModel, qa, variant: str, perm: np.ndarray | None,
                   batch_size: int) -> dict:
    rows = qa["stage_c_row_idx"].to_numpy()
    items = []
    for i, (q, d) in enumerate(zip(qa["question"], qa["delta_str"])):
        prompt_ids, answer_ids = m.prompt_ids(q, d)
        z = m.z_batch(rows[i : i + 1], variant, perm)[0].cpu()
        items.append({"z": z, "prompt_ids": prompt_ids, "answer_ids": answer_ids})
    seq_ce, seq_em = [], []
    for lo in range(0, len(items), batch_size):
        batch = collate_qa(items[lo : lo + batch_size], m.tokenizer.pad_token_id)
        z = batch["z"].to(m.device)
        ids = batch["input_ids"].to(m.device)
        labels = batch["labels"].to(m.device)
        attn = batch["attention_mask"].to(m.device)
        b = ids.shape[0]
        enc = m.encoder.encode(z.unsqueeze(1), out_dtype=m.dtype)
        embeds = m.embed(ids)
        inputs_embeds = torch.cat([enc.soft_tokens, embeds], dim=1)
        full_labels = torch.cat([labels.new_full((b, 1), -100), labels], dim=1)
        full_attn = torch.cat([attn.new_ones((b, 1)), attn], dim=1)
        out = m.decoder(inputs_embeds=inputs_embeds, attention_mask=full_attn,
                        use_cache=False)
        logits = out.logits[:, :-1].float()
        tgt = full_labels[:, 1:]
        ce = F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none",
                             ignore_index=-100)
        sup = (tgt != -100).float()
        seq_ce += ((ce * sup).sum(dim=1) / sup.sum(dim=1)).tolist()
        seq_em += (((logits.argmax(-1) == tgt) | (tgt == -100)).all(dim=-1)).tolist()
    return {"ce": round(float(np.mean(seq_ce)), 4), "em": round(float(np.mean(seq_em)), 4)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=KINDS["pcd"]["run"])
    p.add_argument("--ks", nargs="+", type=int, default=[16, 32, 64])
    p.add_argument("--tf-per-class", type=int, default=128)
    p.add_argument("--gen-per-class", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    m = PCDEvalModel(args.run, eval_device())
    tf_sample = class_sample(load_qa(m.cfg), args.tf_per_class, args.seed)
    gen_sample = class_sample(load_eval_qa(m.cfg), args.gen_per_class, args.seed)
    perm = scramble_perm(len(m.z), args.seed)
    gen_deltas = gen_sample["delta"].to_numpy()

    results = {}
    for k in args.ks:
        m.encoder.k = k
        rec = {
            "real": teacher_forced(m, tf_sample, "real", None, args.batch_size),
            "scramble": teacher_forced(m, tf_sample, "scramble", perm, args.batch_size),
        }
        g = gen_metrics(
            generate_pcd(m, gen_sample, "real", None, args.batch_size,
                         args.max_new_tokens),
            gen_deltas,
        )
        rec["gen_real"] = {
            "sign_acc": round(g["sign_acc"], 4),
            "sign_acc_committed": round(g["sign_acc_committed"], 4),
            "pred_zero_rate": round(g["pred_zero_rate"], 4),
            "mae": round(g["mae"], 1),
            "n": g["n"],
        }
        results[f"k={k}"] = rec
        print(f"[ksweep] k={k} {rec}", flush=True)

    out = {
        "results": results,
        "checkpoint": f"{Path(args.run).name}/final",
        "note": NOTE,
    }
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "artifacts/baselines/evals/eval_e3_ksweep.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[ksweep] -> {out_path}")


if __name__ == "__main__":
    main()
