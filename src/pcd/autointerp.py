from __future__ import annotations

import heapq
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from src.pcd.data import INSTRUCT_PREFIX
from src.pcd.encoder import TopKEncoder
from src.subject.model_io import load_model_and_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# exemplar collection
# --------------------------------------------------------------------------

@torch.no_grad()
def collect_exemplars(
    subject,
    encoder: TopKEncoder,
    passages: list[list[int]],
    read_layer: int,
    device,
    top_n: int = 20,
    ctx_radius: int = 8,
    batch_passages: int = 16,
) -> dict[int, list[dict]]:
    heaps: dict[int, list] = {}
    seq_len = len(passages[0])
    for b0 in range(0, len(passages), batch_passages):
        chunk = passages[b0 : b0 + batch_passages]
        input_ids = torch.tensor(chunk, dtype=torch.long, device=device)
        out = subject(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[read_layer]
        enc = encoder.encode(h.reshape(-1, h.shape[-1]))
        idx = enc.topk_indices.reshape(len(chunk), seq_len, -1)
        val = enc.topk_values.reshape(len(chunk), seq_len, -1)
        idx_cpu, val_cpu = idx.cpu().tolist(), val.cpu().tolist()
        for bi, pid in enumerate(range(b0, b0 + len(chunk))):
            for pos in range(seq_len):
                for cid, v in zip(idx_cpu[bi][pos], val_cpu[bi][pos]):
                    if v <= 0:
                        continue
                    hp = heaps.setdefault(cid, [])
                    entry = (v, pid, pos)
                    if len(hp) < top_n:
                        heapq.heappush(hp, entry)
                    elif v > hp[0][0]:
                        heapq.heapreplace(hp, entry)
    result: dict[int, list[dict]] = {}
    for cid, hp in heaps.items():
        result[cid] = [
            {"value": float(v), "passage": pid, "pos": pos}
            for v, pid, pos in sorted(hp, reverse=True)
        ]
    return result


def exemplar_text(tokenizer, passages: list[list[int]], ex: dict, ctx_radius: int) -> str:
    ids = passages[ex["passage"]]
    pos = ex["pos"]
    lo, hi = max(0, pos - ctx_radius), min(len(ids), pos + ctx_radius + 1)
    before = tokenizer.decode(ids[lo:pos], skip_special_tokens=True,
                              clean_up_tokenization_spaces=False)
    tok = tokenizer.decode([ids[pos]], skip_special_tokens=True,
                           clean_up_tokenization_spaces=False)
    after = tokenizer.decode(ids[pos + 1 : hi], skip_special_tokens=True,
                             clean_up_tokenization_spaces=False)
    return f"{before}«{tok}»{after}".replace("\n", " ").strip()


# --------------------------------------------------------------------------
# LLM backends
# --------------------------------------------------------------------------

class MockBackend:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def describe(self, concept_id: int, exemplars: list[str]) -> str:
        head = exemplars[0][:60] if exemplars else "(no exemplars)"
        return f"[mock] concept {concept_id} fires around: {head}"

    def score(self, concept_id: int, description: str, exemplars: list[str],
              values: list[float]) -> float:
        return round(self.rng.uniform(0.3, 0.9), 4)


class OpenAIBackend:
    def __init__(self, model: str, base_url: str | None = None, temperature: float = 0.0):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url) if base_url else OpenAI()
        self.model = model
        self.temperature = temperature

    def describe(self, concept_id: int, exemplars: list[str]) -> str:
        bullets = "\n".join(f"- {e}" for e in exemplars[:20])
        prompt = (
            "Below are text snippets where a neural-network concept activates; the "
            "activating token is wrapped in \u00ab\u00bb. Snippets are ordered from strongest "
            "to weakest activation.\n\n" + bullets + "\n\n"
            "In one short phrase, describe what the concept fires on. The description "
            "must account for ALL the snippets above (weight the strongest most), so "
            "prefer the most general pattern that fits every snippet over a narrower "
            "pattern that fits only a subset (e.g. if the marked tokens are Fernando, "
            "Frantz, Felipa, say 'first names starting with F', not 'names starting "
            "with Fel').\n\nDescription:"
        )
        r = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content.strip()

    def score(self, concept_id: int, description: str, exemplars: list[str],
              values: list[float]) -> float:
        import numpy as np

        listing = "\n".join(f"{i}. {e}" for i, e in enumerate(exemplars))
        prompt = (
            f"A concept is described as: \"{description}\".\n"
            "For each snippet below, predict how strongly the concept activates on "
            "the «»-marked token, as an integer 0-10. Return one integer per line, "
            "in order, no other text.\n\n" + listing
        )
        r = self.client.chat.completions.create(
            model=self.model, temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        preds = []
        for line in r.choices[0].message.content.splitlines():
            line = line.strip().split(".")[-1].strip()
            try:
                preds.append(float(line))
            except ValueError:
                continue
        n = min(len(preds), len(values))
        if n < 2:
            return 0.0
        p, v = np.array(preds[:n]), np.array(values[:n])
        if p.std() == 0 or v.std() == 0:
            return 0.0
        return float(np.corrcoef(p, v)[0, 1])


def make_backend(cfg: dict):
    kind = cfg.get("backend", "mock")
    if kind == "mock":
        return MockBackend(seed=cfg.get("seed", 0))
    if kind == "openai":
        return OpenAIBackend(model=cfg["model"], base_url=cfg.get("base_url"),
                             temperature=cfg.get("temperature", 0.0))
    raise ValueError(f"unknown auto-interp backend {kind!r}")


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------

def load_corpus(cfg: dict, tokenizer) -> list[list[int]]:
    import pandas as pd

    seq_len = cfg.get("seq_len", 64)
    max_passages = cfg.get("max_passages", 2000)
    prefix_ids = tokenizer(INSTRUCT_PREFIX, add_special_tokens=False)["input_ids"]
    body = seq_len - len(prefix_ids)
    if body <= 0:
        raise ValueError("seq_len too short for the instruct prefix")

    texts: list[str] = []
    src = cfg.get("source", "applications")
    if src == "applications":
        df = pd.read_parquet(REPO_ROOT / cfg.get("path", "data/applications/subject_set.parquet"))
        split = cfg.get("split", "qa_pool")
        if split:
            df = df[df["split_tag"] == split]
        texts = df["application_text"].head(max_passages * 2).tolist()
    elif src == "synthetic":
        from src.pcd.data import iter_synthetic

        it = iter_synthetic({}, 0, 1)
        texts = [next(it) for _ in range(max_passages * 2)]
    else:
        raise ValueError(f"unknown corpus source {src!r}")

    passages: list[list[int]] = []
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False)["input_ids"][:body]
        if len(ids) < body:
            continue
        passages.append(prefix_ids + ids)
        if len(passages) >= max_passages:
            break
    if not passages:
        raise ValueError("empty auto-interp corpus after tokenization")
    return passages


# --------------------------------------------------------------------------
# entry
# --------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def apply_minimal_run(cfg: dict) -> dict:
    a = cfg.setdefault("autointerp", {})
    a["max_passages"] = min(a.get("max_passages") or 64, 64)
    a["seq_len"] = min(a.get("seq_len") or 48, 48)
    a["n_concepts_to_label"] = min(a.get("n_concepts_to_label") or 32, 32)
    a["top_n"] = min(a.get("top_n") or 10, 10)
    a.setdefault("corpus", {})["source"] = "synthetic"
    a.setdefault("llm", {})["backend"] = "mock"
    cfg.setdefault("run", {})
    cfg["run"]["name"] = (cfg["run"].get("name") or "autointerp") + "-minimal"
    cfg["minimal_run"] = True
    return cfg


def run(config_path, minimal_run: bool = False) -> dict:
    cfg = load_config(config_path)
    if minimal_run or cfg.get("minimal_run"):
        cfg = apply_minimal_run(cfg)
    a = cfg["autointerp"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    read_layer = cfg.get("read_layer", 15)

    run_name = cfg.get("run", {}).get("name") or "autointerp"
    out_dir = REPO_ROOT / a.get("output_root", "artifacts/pcd") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[autointerp] run dir: {out_dir}")

    # ---- subject + frozen encoder ----
    subject, tokenizer, repo = load_model_and_tokenizer(cfg["model"])
    if a.get("subject_adapter"):
        from peft import PeftModel

        subject = PeftModel.from_pretrained(subject, a["subject_adapter"]).merge_and_unload()
    subject.to(device).eval().requires_grad_(False)

    encoder = TopKEncoder(
        d_model=subject.config.hidden_size,
        n_concepts=cfg["encoder"].get("n_concepts", 32768),
        k=cfg["encoder"].get("k", 16),
    ).to(device)
    ckpt = a.get("encoder_ckpt")
    if ckpt:
        encoder.load_state_dict(torch.load(REPO_ROOT / ckpt, map_location=device))
        print(f"[autointerp] loaded encoder {ckpt}")
    else:
        print("[autointerp] WARNING: no encoder_ckpt; using randomly-initialized encoder")
    encoder.eval()

    # ---- corpus + exemplars ----
    corpus_cfg = {**a.get("corpus", {}), "seq_len": a.get("seq_len", 64),
                  "max_passages": a.get("max_passages", 2000)}
    passages = load_corpus(corpus_cfg, tokenizer)
    t0 = time.time()
    exemplars = collect_exemplars(
        subject, encoder, passages, read_layer, device,
        top_n=a.get("top_n", 20), ctx_radius=a.get("ctx_radius", 8),
        batch_passages=a.get("batch_passages", 16),
    )
    print(f"[autointerp] {len(exemplars)} live concepts over {len(passages)} passages "
          f"({time.time()-t0:.1f}s)")

    # ---- select concepts to label ----
    rng = random.Random(a.get("seed", 0))
    min_ex = a.get("min_exemplars", 1)
    live = sorted(c for c, exs in exemplars.items() if len(exs) >= min_ex)
    print(f"[autointerp] {len(live)}/{len(exemplars)} live concepts have >= {min_ex} exemplars")
    n_label = min(a.get("n_concepts_to_label", 400), len(live))
    chosen = sorted(rng.sample(live, n_label)) if n_label < len(live) else live

    # ---- describe + score ----
    backend = make_backend(a.get("llm", {"backend": "mock"}))
    ctx_radius = a.get("ctx_radius", 8)
    seq_len = len(passages[0])
    labels: dict[str, Any] = {}
    scores = []
    progress_path = out_dir / "progress.log"
    t_label = time.time()
    for i_c, cid in enumerate(chosen):
        exs = exemplars[cid]
        texts = [exemplar_text(tokenizer, passages, e, ctx_radius) for e in exs]
        values = [e["value"] for e in exs]
        n_desc = a.get("n_describe", max(1, len(texts) // 2))
        desc = backend.describe(cid, texts[:n_desc])
        pos_texts = texts[n_desc:] or texts
        pos_values = values[n_desc:] or values
        fired = {(e["passage"], e["pos"]) for e in exs}
        neg = []
        while len(neg) < max(len(pos_texts), 8):
            pid, pos = rng.randrange(len(passages)), rng.randrange(seq_len)
            if (pid, pos) not in fired:
                neg.append({"value": 0.0, "passage": pid, "pos": pos})
        mixed = ([(t, v) for t, v in zip(pos_texts, pos_values)]
                 + [(exemplar_text(tokenizer, passages, e, ctx_radius), 0.0) for e in neg])
        rng.shuffle(mixed)
        score = backend.score(cid, desc, [t for t, _ in mixed], [v for _, v in mixed])
        labels[str(cid)] = {
            "description": desc,
            "auto_interp_score": score,
            "n_exemplars": len(exs),
            "max_activation": values[0] if values else 0.0,
            "top_exemplars": texts[:5],
        }
        scores.append(score)
        if (i_c + 1) % 5 == 0 or (i_c + 1) == len(chosen):
            rate = (i_c + 1) / max(time.time() - t_label, 1e-9)
            eta = (len(chosen) - i_c - 1) / rate
            line = (f"{time.strftime('%H:%M:%S')} labeled {i_c + 1}/{len(chosen)} "
                    f"| mean r so far {sum(scores)/len(scores):.3f} "
                    f"| {rate*60:.1f} concepts/min | ETA {eta/60:.1f} min")
            print(f"[autointerp] {line}", flush=True)
            with open(progress_path, "a") as f:
                f.write(line + "\n")

    labels_path = out_dir / "concept_labels.json"
    labels_path.write_text(json.dumps(labels, indent=2))
    report = {
        "run_name": run_name, "model_repo": repo,
        "encoder_ckpt": ckpt, "n_passages": len(passages),
        "n_live_concepts": len(exemplars),
        "n_concepts_labeled": len(chosen),
        "mean_auto_interp_score": round(sum(scores) / len(scores), 4) if scores else None,
        "backend": a.get("llm", {}).get("backend", "mock"),
        "seconds": round(time.time() - t0, 1),
        "labels_path": str(labels_path.relative_to(REPO_ROOT)),
        "minimal_run": bool(cfg.get("minimal_run")),
    }
    (out_dir / "autointerp_report.json").write_text(json.dumps(report, indent=2))
    print("[autointerp] report: " + json.dumps(report, indent=2))
    return report
