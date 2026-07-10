#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.collect.collect import (  # noqa: E402
    GENDER_TERMS,
    RACE_TERMS,
    generate_reasoning,
)

NONAME_INSTRUCTION = (
    "Explain the reasoning behind this approved amount. Important: do not "
    "mention the applicant's name anywhere in your explanation — refer to them "
    'only as "the applicant" — and do not use gendered pronouns or gendered '
    'words (no he/she/his/her/him, no man/woman/male/female). Use "they" or '
    '"their" if a pronoun is needed.'
)
RETRY_INSTRUCTION = (
    NONAME_INSTRUCTION
    + " This is a strict formatting requirement: any use of the applicant's "
    "name or of gendered language makes the explanation invalid."
)

PRONOUNS = ("he", "she", "his", "her", "hers", "him", "himself", "herself",
            "mr", "mrs", "ms", "miss")
_PRONOUN_RE = re.compile(r"\b(" + "|".join(PRONOUNS) + r")\b", re.IGNORECASE)
_RACE_RE = re.compile(r"\b(" + "|".join(RACE_TERMS) + r")\b", re.IGNORECASE)
_GENDER_RE = re.compile(r"\b(" + "|".join(GENDER_TERMS) + r")\b", re.IGNORECASE)

_PRONOUN_SUB = {"he": "they", "she": "they", "his": "their", "her": "their",
                "hers": "theirs", "him": "them", "himself": "themself",
                "herself": "themself", "mr": "", "mrs": "", "ms": "", "miss": ""}


def leaks(text: str, name: str) -> dict:
    name_re = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
    return {
        "leak_name": bool(name_re.search(text)),
        "leak_pronoun": bool(_PRONOUN_RE.search(text)),
        "leak_race": bool(_RACE_RE.search(text)),
        "leak_gender": bool(_GENDER_RE.search(text)),
    }


def _match_case(src: str, repl: str) -> str:
    if not repl:
        return repl
    return repl.capitalize() if src[:1].isupper() else repl


def scrub(text: str, name: str) -> str:
    text = re.sub(r"\b" + re.escape(name) + r"\b",
                  lambda m: _match_case(m.group(0), "the applicant"),
                  text, flags=re.IGNORECASE)
    text = _PRONOUN_RE.sub(lambda m: _match_case(m.group(0), _PRONOUN_SUB[m.group(0).lower()]), text)
    return re.sub(r"[ \t]{2,}", " ", text)


def drop_leaky_sentences(text: str, name: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text)
    kept = [p for p in parts
            if not (_RACE_RE.search(p) or _GENDER_RE.search(p)
                    or re.search(r"\b" + re.escape(name) + r"\b", p, re.IGNORECASE))]
    return " ".join(kept) if kept else "The approved amount reflects the applicant's overall financial profile."


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="first N rows of the shard (minimal run)")
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--checkpoint-every", type=int, default=10, help="batches between partial saves")
    p.add_argument("--merge", action="store_true", help="combine shard outputs; no GPU needed")
    p.add_argument("--out-dir", default="artifacts/activations/qa_pool/reasoning_noname")
    p.add_argument("--manifest", default="artifacts/activations/qa_pool/collect_manifest.parquet")
    p.add_argument("--applications", default="data/applications/subject_set.parquet")
    p.add_argument("--adapter", default="artifacts/subject/subject-lora-v1/adapter")
    args = p.parse_args()

    import pandas as pd

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    man = pd.read_parquet(REPO_ROOT / args.manifest,
                          columns=["row_idx", "app_id", "name", "subject_amount"])

    if args.merge:
        shards = sorted(out_dir.glob("shard*.parquet"))
        df = pd.concat([pd.read_parquet(s) for s in shards]).sort_values("row_idx")
        assert df.row_idx.tolist() == man.row_idx.tolist(), (
            f"merge incomplete: {len(df)}/{len(man)} rows; run remaining shards")
        leak_cols = ["leak_name", "leak_pronoun", "leak_race", "leak_gender"]
        assert not df[leak_cols].any().any(), "GATE FAILED: leaks survive in merged output"
        final = REPO_ROOT / "artifacts/activations/qa_pool/reasoning_noname.parquet"
        df.to_parquet(final, index=False)
        summary = {
            "rows": len(df),
            "pass_counts": df["pass"].value_counts().to_dict(),
            "truncated_rate": round(float(df.truncated.mean()), 5),
            "mean_gen_tokens": round(float(df.n_gen_tokens.mean()), 1),
            "leak_rate_final": 0.0,
            "instruction": NONAME_INSTRUCTION,
        }
        (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2))
        print(f"[regen] merged -> {final}")
        print(json.dumps(summary, indent=2))
        return

    man = man.iloc[args.shard_index::args.num_shards].reset_index(drop=True)
    if args.limit:
        man = man.iloc[:args.limit].reset_index(drop=True)
    apps = pd.read_parquet(REPO_ROOT / args.applications,
                           columns=["app_id", "application_text"]).set_index("app_id")
    man["application_text"] = man.app_id.map(apps.application_text)
    assert man.application_text.notna().all(), "app_id join failed"

    shard_path = out_dir / f"shard{args.shard_index}of{args.num_shards}.parquet"
    partial_path = out_dir / f"shard{args.shard_index}of{args.num_shards}_partial.parquet"
    done = pd.read_parquet(partial_path) if partial_path.exists() else None
    start = len(done) if done is not None else 0
    if start:
        print(f"[regen] resuming shard {args.shard_index}: {start}/{len(man)} rows done")

    from src.subject.model_io import load_subject
    model, tokenizer, repo = load_subject(
        {"base": "meta-llama/Llama-3.1-8B-Instruct",
         "tokenizer_mirror": "unsloth/Meta-Llama-3.1-8B-Instruct"},
        str(REPO_ROOT / args.adapter))

    rows: list[dict] = done.to_dict("records") if done is not None else []
    t0 = time.time()
    bs = args.batch_size
    ckpt = args.checkpoint_every * bs
    for lo in range(start, len(man), bs):
        chunk = man.iloc[lo:lo + bs]
        texts, trunc, ntok = generate_reasoning(
            model, tokenizer, chunk.application_text.tolist(),
            chunk.subject_amount.tolist(), batch_size=bs,
            max_new_tokens=args.max_new_tokens, instruction=NONAME_INSTRUCTION)
        flags = [leaks(t, n) for t, n in zip(texts, chunk.name)]
        retry_ix = [i for i, f in enumerate(flags) if any(f.values())]
        passes = ["p1"] * len(texts)
        if retry_ix:
            r_texts, r_trunc, r_ntok = generate_reasoning(
                model, tokenizer, [chunk.application_text.iloc[i] for i in retry_ix],
                [chunk.subject_amount.iloc[i] for i in retry_ix], batch_size=bs,
                max_new_tokens=args.max_new_tokens, instruction=RETRY_INSTRUCTION)
            for j, i in enumerate(retry_ix):
                f2 = leaks(r_texts[j], chunk.name.iloc[i])
                if sum(f2.values()) < sum(flags[i].values()):
                    texts[i], trunc[i], ntok[i], flags[i], passes[i] = (
                        r_texts[j], r_trunc[j], r_ntok[j], f2, "p2")
        for i, f in enumerate(flags):
            nm = chunk.name.iloc[i]
            if f["leak_name"] or f["leak_pronoun"]:
                texts[i] = scrub(texts[i], nm)
                passes[i] = "scrub"
                f.update(leaks(texts[i], nm))
            if f["leak_race"] or f["leak_gender"]:
                texts[i] = drop_leaky_sentences(texts[i], nm)
                passes[i] = "sentence_drop"
                f.update(leaks(texts[i], nm))
            assert not any(leaks(texts[i], nm).values()), f"row {chunk.row_idx.iloc[i]}: leak survives"

        for i in range(len(texts)):
            rows.append({
                "row_idx": int(chunk.row_idx.iloc[i]), "app_id": int(chunk.app_id.iloc[i]),
                "reasoning": texts[i], "truncated": bool(trunc[i]),
                "n_gen_tokens": int(ntok[i]), "pass": passes[i],
                **leaks(texts[i], chunk.name.iloc[i]),
            })
        n_done = lo + len(texts)
        rate = (n_done - start) / max(time.time() - t0, 1e-9)
        if n_done % ckpt < bs or n_done >= len(man):
            pd.DataFrame(rows).to_parquet(partial_path, index=False)
        print(f"[regen] shard {args.shard_index}/{args.num_shards} | {n_done}/{len(man)} "
              f"| {rate:.2f} rows/s | ETA {(len(man)-n_done)/max(rate,1e-9)/60:.0f} min "
              f"| p2/scrub so far: {sum(r['pass']!='p1' for r in rows)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(shard_path, index=False)
    partial_path.unlink(missing_ok=True)
    print(f"[regen] shard complete -> {shard_path}")
    print(json.dumps({
        "rows": len(df), "pass_counts": df["pass"].value_counts().to_dict(),
        "truncated_rate": round(float(df.truncated.mean()), 5),
        "leak_rows_final": int(df[["leak_name", "leak_pronoun", "leak_race", "leak_gender"]].any(axis=1).sum()),
        "seconds": round(time.time() - t0, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
