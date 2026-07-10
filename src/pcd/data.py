from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

REPO_ROOT = Path(__file__).resolve().parents[2]

INSTRUCT_PREFIX = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
)

SOURCE_IDS = {"general": 0, "domain": 1, "application": 2}

_STRONG_FINANCE = (
    r"loans?", r"mortgages?", r"borrowers?", r"lenders?", r"underwrit\w*",
    r"creditworth\w*", r"refinanc\w*", r"amorti[sz]\w*", r"debt[- ]to[- ]income",
    r"line of credit", r"credit scores?", r"credit unions?", r"home equity",
    r"(?:auto|personal|student|payday|business|home|car) loans?",
    r"annual percentage rate", r"repayments?", r"loan amounts?", r"loan-to-value",
    r"foreclosures?", r"loan applications?", r"credit history",
)
_SUPPORT_FINANCE = (
    r"credit", r"interest rates?", r"principal", r"collateral", r"debts?",
    r"delinquen\w*", r"instal?lments?", r"lending", r"monthly payments?",
    r"down payment", r"defaults?", r"income", r"banks?", r"banking", r"escrow",
    r"fico", r"apr",
)
_STRONG_RE = re.compile(r"\b(?:" + "|".join(_STRONG_FINANCE) + r")\b")
_SUPPORT_RE = re.compile(r"\b(?:" + "|".join(_SUPPORT_FINANCE) + r")\b")
FINANCE_MIN_STRONG = 3
FINANCE_MIN_DISTINCT = 2


def is_finance(text: str, min_strong: int = FINANCE_MIN_STRONG,
               min_distinct: int = FINANCE_MIN_DISTINCT) -> bool:
    low = text.lower()
    strong_matches = _STRONG_RE.findall(low)
    if len(strong_matches) < min_strong:
        return False
    return len(set(strong_matches) | set(_SUPPORT_RE.findall(low))) >= min_distinct


# --------------------------------------------------------------------------
# pure helpers (unit-tested)
# --------------------------------------------------------------------------

@dataclass
class Segments:
    prefix: list[int]
    middle: list[int]
    suffix: list[int]
    suffix_mask: list[int]
    source: str


def chunk_window(
    ids: list[int],
    n_prefix: int,
    n_middle: int,
    n_suffix: int,
    rng: random.Random,
    window: str = "random",
) -> tuple[int, list[int]] | None:
    total = n_prefix + n_middle + n_suffix
    if len(ids) < total:
        return None
    max_start = len(ids) - total
    start = 0 if window == "head" or max_start == 0 else rng.randint(0, max_start)
    return start, ids[start : start + total]


def split_segments(
    window_ids: list[int],
    n_prefix: int,
    n_middle: int,
    n_suffix: int,
    source: str,
    mask_positions: set[int] | None = None,
    window_start: int = 0,
) -> Segments:
    prefix = window_ids[:n_prefix]
    middle = window_ids[n_prefix : n_prefix + n_middle]
    suffix = window_ids[n_prefix + n_middle : n_prefix + n_middle + n_suffix]
    suffix_start_global = window_start + n_prefix + n_middle
    if mask_positions:
        suffix_mask = [
            0 if (suffix_start_global + j) in mask_positions else 1
            for j in range(len(suffix))
        ]
    else:
        suffix_mask = [1] * len(suffix)
    return Segments(prefix, middle, suffix, suffix_mask, source)


def decision_line_positions(app_ids: list[int], anchor_token_id: int, anchor_len: int) -> set[int]:
    try:
        anchor_pos = len(app_ids) - 1 - app_ids[::-1].index(anchor_token_id)
    except ValueError:
        return set()
    start = max(0, anchor_pos - (anchor_len - 1))
    return set(range(start, len(app_ids)))


# --------------------------------------------------------------------------
# raw text sources
# --------------------------------------------------------------------------

def iter_fineweb(cfg: dict, shard_index: int, num_shards: int) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset(
        cfg.get("hf_name", "HuggingFaceFW/fineweb"),
        name=cfg.get("hf_subset", "sample-10BT"),
        split=cfg.get("hf_split", "train"),
        streaming=True,
    )
    for i, row in enumerate(ds):
        if num_shards > 1 and (i % num_shards) != shard_index:
            continue
        text = row.get("text") or ""
        if len(text) >= cfg.get("min_chars", 400):
            yield text


def iter_local(cfg: dict, shard_index: int, num_shards: int) -> Iterator[str]:
    path = REPO_ROOT / cfg["local_path"]
    files = sorted(path.glob("**/*.txt")) if path.is_dir() else [path]
    passages: list[str] = []
    for f in files:
        for para in f.read_text().split("\n\n"):
            para = para.strip()
            if len(para) >= cfg.get("min_chars", 200):
                passages.append(para)
    if not passages:
        raise ValueError(f"no passages >= min_chars in {path}")
    i = 0
    while True:
        if num_shards <= 1 or (i % num_shards) == shard_index:
            yield passages[i % len(passages)]
        i += 1


_SYNTH_WORDS = (
    "the credit application review noted stable income and moderate debt while "
    "the lender assessed repayment history against the requested principal and "
    "interest terms before underwriting a final decision about the borrower "
    "account balance employment tenure and external risk scores summary report"
).split()


def iter_synthetic(cfg: dict, shard_index: int, num_shards: int, seed: int = 0) -> Iterator[str]:
    rng = random.Random(seed * 100003 + shard_index)
    i = 0
    while True:
        n = rng.randint(60, 120)
        text = " ".join(rng.choice(_SYNTH_WORDS) for _ in range(n))
        if num_shards <= 1 or (i % num_shards) == shard_index:
            yield text
        i += 1


def make_text_source(cfg: dict, shard_index: int, num_shards: int, seed: int) -> Iterator[str]:
    src = cfg.get("source", "fineweb")
    if src == "fineweb":
        return iter_fineweb(cfg, shard_index, num_shards)
    if src == "cache":
        return iter_cache(cfg, shard_index, num_shards)
    if src == "local":
        return iter_local(cfg, shard_index, num_shards)
    if src == "synthetic":
        return iter_synthetic(cfg, shard_index, num_shards, seed)
    raise ValueError(f"unknown text source {src!r}")


def iter_cache(cfg: dict, shard_index: int, num_shards: int) -> Iterator[str]:
    import pandas as pd

    texts = pd.read_parquet(REPO_ROOT / cfg["cache_file"], columns=["text"])["text"].tolist()
    if not texts:
        raise ValueError(f"empty cache {cfg['cache_file']}")
    i = 0
    while True:
        if num_shards <= 1 or (i % num_shards) == shard_index:
            yield texts[i % len(texts)]
        i += 1


def load_application_texts(cfg: dict) -> list[str]:
    import pandas as pd

    df = pd.read_parquet(REPO_ROOT / cfg["path"])
    split = cfg.get("split", "subject_train")
    if split:
        df = df[df["split_tag"] == split]
    name_split = cfg.get("name_split", "pcd_train")
    nsp_path = REPO_ROOT / cfg.get("name_split_path", "data/names/name_splits.parquet")
    if name_split:
        if not nsp_path.exists():
            raise FileNotFoundError(
                f"name_split={name_split!r} requested but {nsp_path} missing; "
                "set data.application.name_split: null to disable the guard."
            )
        nsp = pd.read_parquet(nsp_path)[["name", "name_split"]]
        df = df.merge(nsp, on="name", how="left", validate="many_to_one")
        df = df[df["name_split"] == name_split]
    exclude = cfg.get("exclude_races") or []
    if exclude:
        df = df[~df["name_race"].isin(exclude)]
    if len(df) == 0:
        raise ValueError("application slice is empty after filtering")
    append = cfg.get("append_amount", True)
    if append:
        return [t + str(int(a)) for t, a in zip(df["application_text"], df["amount"])]
    return df["application_text"].tolist()


def build_cache(cfg: dict, out_dir: str, n_domain_docs: int, n_general_docs: int = 0,
                max_stream: int | None = None, log_every: int = 100000) -> dict:
    import json
    import time

    import pandas as pd

    src_cfg = cfg["data"]["general"]
    min_chars = src_cfg.get("min_chars", 400)
    it = make_text_source({**src_cfg, "source": src_cfg.get("source", "fineweb")}, 0, 1,
                          cfg.get("train", {}).get("seed", 0))

    domain: list[str] = []
    general: list[str] = []
    n_seen = 0
    t0 = time.time()
    for text in it:
        n_seen += 1
        if len(text) < min_chars:
            continue
        if len(domain) < n_domain_docs and is_finance(text):
            domain.append(text)
        elif len(general) < n_general_docs:
            general.append(text)
        if len(domain) >= n_domain_docs and len(general) >= n_general_docs:
            break
        if max_stream and n_seen >= max_stream:
            break
        if n_seen % log_every == 0:
            print(f"[build_cache] seen {n_seen:,} | domain {len(domain):,}/{n_domain_docs:,} "
                  f"| general {len(general):,}/{n_general_docs:,} | {n_seen/(time.time()-t0):.0f} docs/s",
                  flush=True)

    out = REPO_ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text": domain}).to_parquet(out / "domain.parquet", index=False)
    if general:
        pd.DataFrame({"text": general}).to_parquet(out / "general.parquet", index=False)
    manifest = {
        "source": src_cfg.get("source", "fineweb"),
        "hf_name": src_cfg.get("hf_name"),
        "hf_subset": src_cfg.get("hf_subset"),
        "docs_streamed": n_seen,
        "domain_docs": len(domain),
        "general_docs": len(general),
        "finance_hit_rate": round(len(domain) / n_seen, 5) if n_seen else None,
        "min_chars": min_chars,
        "finance_min_strong": FINANCE_MIN_STRONG,
        "finance_min_distinct": FINANCE_MIN_DISTINCT,
        "seconds": round(time.time() - t0, 1),
        "domain_sample": [t[:200].replace("\n", " ") for t in domain[:8]],
    }
    (out / "cache_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[build_cache] wrote {len(domain):,} domain + {len(general):,} general docs to {out}")
    print(json.dumps({k: v for k, v in manifest.items() if k != "domain_sample"}, indent=2))
    return manifest


# --------------------------------------------------------------------------
# mixture dataset
# --------------------------------------------------------------------------

class PretrainMixture(IterableDataset):
    def __init__(self, cfg: dict, tokenizer, seed: int = 0):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.seed = seed
        seq = cfg.get("seq", {})
        self.n_prefix = seq.get("n_prefix", 16)
        self.n_middle = seq.get("n_middle", 16)
        self.n_suffix = seq.get("n_suffix", 16)
        self.window = cfg.get("window", "random")
        mix = cfg.get("mixture", {"general": 0.72, "domain": 0.18, "application": 0.10})
        self.mix_names = list(mix.keys())
        self.mix_weights = [mix[n] for n in self.mix_names]
        self.anchor_token_id = cfg.get("anchor_token_id", 400)
        self._app_texts: list[str] | None = None
        self._anchor_len: int | None = None

    def _apps(self) -> list[str]:
        if self._app_texts is None:
            if "application" in self.mix_names and self.mix_weights[self.mix_names.index("application")] > 0:
                self._app_texts = load_application_texts(self.cfg["application"])
                self._anchor_len = len(
                    self.tokenizer(self.cfg.get("anchor_text", "AMOUNT APPROVED: $"),
                                   add_special_tokens=False)["input_ids"]
                )
            else:
                self._app_texts = []
                self._anchor_len = 0
        return self._app_texts

    def _shard(self) -> tuple[int, int]:
        rank = int(self.cfg.get("_rank", 0))
        world = int(self.cfg.get("_world_size", 1))
        info = get_worker_info()
        wid = info.id if info else 0
        nworkers = info.num_workers if info else 1
        return rank * nworkers + wid, world * nworkers

    def __iter__(self) -> Iterator[dict]:
        shard_index, num_shards = self._shard()
        rng = random.Random(self.seed * 7919 + shard_index)
        apps = self._apps()

        general_cfg = self.cfg.get("general", {"source": "synthetic"})
        gen_iter = make_text_source(general_cfg, shard_index, num_shards, self.seed)
        domain_cfg = {**general_cfg, **self.cfg.get("domain", {})}
        domain_prefiltered = domain_cfg.get("source") == "cache"
        dom_iter = make_text_source(domain_cfg, shard_index, num_shards, self.seed + 1)
        app_i = shard_index % max(len(apps), 1)

        def next_general() -> Segments | None:
            text = next(gen_iter)
            return self._tokenize_passage(text, "general", rng)

        def next_domain() -> Segments | None:
            if domain_prefiltered:
                return self._tokenize_passage(next(dom_iter), "domain", rng)
            for _ in range(self.cfg.get("domain_reject_limit", 200)):
                text = next(dom_iter)
                if is_finance(text):
                    return self._tokenize_passage(text, "domain", rng)
            return None

        def next_application() -> Segments | None:
            nonlocal app_i
            if not apps:
                return None
            text = apps[app_i % len(apps)]
            app_i += num_shards
            return self._tokenize_app(text, rng)

        makers = {"general": next_general, "domain": next_domain, "application": next_application}
        while True:
            name = rng.choices(self.mix_names, weights=self.mix_weights, k=1)[0]
            seg = makers[name]()
            if seg is None:
                continue
            yield {
                "prefix": seg.prefix,
                "middle": seg.middle,
                "suffix": seg.suffix,
                "suffix_mask": seg.suffix_mask,
                "source": SOURCE_IDS[seg.source],
            }

    def _tokenize_passage(self, text: str, source: str, rng: random.Random) -> Segments | None:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        win = chunk_window(ids, self.n_prefix, self.n_middle, self.n_suffix, rng, self.window)
        if win is None:
            return None
        start, window_ids = win
        return split_segments(window_ids, self.n_prefix, self.n_middle, self.n_suffix, source)

    def _tokenize_app(self, text: str, rng: random.Random) -> Segments | None:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        win = chunk_window(ids, self.n_prefix, self.n_middle, self.n_suffix, rng, self.window)
        if win is None:
            return None
        start, window_ids = win
        mask_positions = None
        if self.cfg["application"].get("mask_decision_line", True):
            mask_positions = decision_line_positions(ids, self.anchor_token_id, self._anchor_len or 1)
        return split_segments(
            window_ids, self.n_prefix, self.n_middle, self.n_suffix,
            "application", mask_positions=mask_positions, window_start=start,
        )


@dataclass
class PretrainCollator:
    instruct_prefix_ids: list[int]
    n_middle: int

    def __call__(self, batch: list[dict]) -> dict:
        subj, suf, lab, src = [], [], [], []
        for ex in batch:
            subj.append(self.instruct_prefix_ids + ex["prefix"] + ex["middle"])
            suf.append(ex["suffix"])
            lab.append([s if m else -100 for s, m in zip(ex["suffix"], ex["suffix_mask"])])
            src.append(ex["source"])
        return {
            "subject_input_ids": torch.tensor(subj, dtype=torch.long),
            "suffix_ids": torch.tensor(suf, dtype=torch.long),
            "suffix_labels": torch.tensor(lab, dtype=torch.long),
            "source": torch.tensor(src, dtype=torch.long),
        }


def build_dataloader(cfg: dict, tokenizer, batch_size: int, rank: int, world_size: int, seed: int = 0):
    from torch.utils.data import DataLoader

    dcfg = dict(cfg)
    dcfg["_rank"] = rank
    dcfg["_world_size"] = world_size
    ds = PretrainMixture(dcfg, tokenizer, seed=seed)
    instruct_ids = (
        tokenizer(INSTRUCT_PREFIX, add_special_tokens=False)["input_ids"]
        if cfg.get("instruct_prefix", True)
        else [tokenizer.bos_token_id]
    )
    collate = PretrainCollator(instruct_ids, cfg.get("seq", {}).get("n_middle", 16))
    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.get("num_workers", 2) > 0,
    )
