from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from numpy.lib.format import open_memmap

from src.cf_dataset import questions as Q
from src.cf_dataset.splits import Holdout, assign_split, coarse_split
from src.subject.decode_eval import decode_amounts
from src.subject.model_io import load_subject
from src.subject.probe import collect_anchor_activations

REPO_ROOT = Path(__file__).resolve().parents[2]
CELLS = [
    "white_male", "white_female", "asian_male", "asian_female",
    "hispanic_male", "hispanic_female", "black_male", "black_female",
]


# --------------------------------------------------------------------------
# per-app instance planning
# --------------------------------------------------------------------------

class NameSampler:
    def __init__(self, name_sample: pd.DataFrame, name_splits: pd.DataFrame):
        df = name_sample[["name", "race", "gender", "cell"]].merge(
            name_splits[["name", "name_split"]], on="name", how="left"
        )
        self.by_name = df.set_index("name")
        self.heldout_names = set(df.loc[df.name_split == "pcd_heldout", "name"])
        self._cell_all = {c: df.loc[df.cell == c, "name"].to_numpy() for c in CELLS}
        self._cell_train = {
            c: df.loc[(df.cell == c) & (df.name_split == "pcd_train"), "name"].to_numpy()
            for c in CELLS
        }
        self._all_names = df["name"].to_numpy()

    def point_targets(self, current_name: str, n: int, rng) -> list[dict]:
        picks, seen = [], {current_name}
        while len(picks) < n:
            nm = str(rng.choice(self._all_names))
            if nm in seen:
                continue
            seen.add(nm)
            row = self.by_name.loc[nm]
            picks.append({"target_name": nm, "target_race": row["race"],
                          "target_gender": row["gender"], "target_cell": row["cell"]})
        return picks

    def mc_names(self, cell: str, m: int, eval_instance: bool, rng) -> list[str]:
        pool = self._cell_all[cell] if eval_instance else self._cell_train[cell]
        if len(pool) == 0:
            pool = self._cell_all[cell]
        return [str(x) for x in rng.choice(pool, size=m, replace=len(pool) < m)]


def plan_app(row, current_name, sampler, holdout, app_is_sense1, cfg, rng):
    out = []
    for inst in Q.quant_instances(row, current_name):
        out.append(_finalize(inst, "quant", inst.target_meta, app_is_sense1,
                             holdout, sampler.heldout_names))
    for inst in Q.categorical_instances(row, current_name):
        out.append(_finalize(inst, "categorical", inst.target_meta, app_is_sense1,
                             holdout, sampler.heldout_names))
    for tgt in sampler.point_targets(current_name, cfg["n_name_point"], rng):
        inst = Q.name_point_instance(row, tgt["target_name"], tgt)
        meta = {"target_name": tgt["target_name"], "target_race": tgt["target_race"],
                "cell": tgt["target_cell"]}
        out.append(_finalize(inst, "name_point", meta, app_is_sense1,
                             holdout, sampler.heldout_names))
    cells = list(rng.choice(CELLS, size=cfg["n_name_categorical"], replace=False))
    for cell in cells:
        race, gender = cell.rsplit("_", 1)
        meta = {"target_race": race, "target_gender": gender, "cell": cell}
        split = assign_split("name_categorical", meta, app_is_sense1, holdout,
                             sampler.heldout_names)
        is_eval = coarse_split(split) != "pcd_train"
        mc = sampler.mc_names(cell, cfg["mc_m"], is_eval, rng)
        out.append({
            "question_class": "name_categorical",
            "phrase": Q.name_categorical_phrase(race, gender),
            "question": Q.question_text(Q.name_categorical_phrase(race, gender)),
            "is_point": False, "split_fine": split, "split": coarse_split(split),
            "store_z": False, "target_meta": meta,
            "mc_texts": [Q.name_categorical_cf_text(row, nm) for nm in mc],
        })
    return out


def _finalize(inst, qclass, meta, app_is_sense1, holdout, heldout_names):
    split = assign_split(qclass, meta, app_is_sense1, holdout, heldout_names)
    coarse = coarse_split(split)
    return {
        "question_class": qclass, "phrase": inst.phrase, "question": inst.question,
        "cf_text": inst.cf_text, "is_point": True, "split_fine": split,
        "split": coarse, "store_z": coarse != "pcd_train", "target_meta": meta,
    }


# --------------------------------------------------------------------------
# entry
# --------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def apply_minimal_run(cfg):
    cfg["data"]["max_apps"] = min(cfg["data"].get("max_apps") or 24, 24)
    cfg["build"]["chunk_apps"] = 24
    cfg["build"]["mc_m"] = 4
    cfg.setdefault("run", {})
    cfg["run"]["name"] = (cfg["run"].get("name") or "cfqa") + "-minimal"
    cfg["minimal_run"] = True
    return cfg


def run(config_path, minimal_run=False):
    cfg = load_config(config_path)
    if minimal_run or cfg.get("minimal_run"):
        cfg = apply_minimal_run(cfg)
    d, b = cfg["data"], cfg["build"]
    anchor_id, read_layer = cfg["anchor"]["token_id"], cfg["read_layer"]
    seed = b.get("seed", 41)

    run_name = cfg.get("run", {}).get("name") or "cfqa"
    art_dir = REPO_ROOT / b.get("art_root", "artifacts/counterfactuals") / run_name
    data_dir = REPO_ROOT / b.get("data_root", "data/counterfactual_qa") / run_name
    art_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cfqa] art={art_dir} data={data_dir}")

    # ---- data ----
    apps = pd.read_parquet(REPO_ROOT / d["path"])
    apps = apps[apps["split_tag"] == d.get("split", "qa_pool")].reset_index(drop=True)
    man = pd.read_parquet(REPO_ROOT / d["manifest"])[["app_id", "row_idx", "subject_amount"]]
    apps = apps.merge(man, on="app_id", how="inner", validate="one_to_one")
    apps = apps[apps["subject_amount"].notna()].reset_index(drop=True)
    if d.get("max_apps"):
        apps = apps.head(d["max_apps"]).reset_index(drop=True)
    n_apps = len(apps)

    rng0 = np.random.default_rng(seed)
    n_s1 = int(round(b.get("sense1_app_frac", 0.10) * n_apps))
    s1_apps = set(rng0.choice(apps["app_id"].to_numpy(), size=n_s1, replace=False).tolist())

    sampler = NameSampler(
        pd.read_parquet(REPO_ROOT / "data/names/name_sample.parquet"),
        pd.read_parquet(REPO_ROOT / "data/names/name_splits.parquet"),
    )
    holdout = Holdout()

    model, tokenizer, repo = load_subject(cfg["model"], b["adapter"])
    dim = model.config.hidden_size

    ub = n_apps * (len(Q.DEFAULT_QUANT) + len(Q.DEFAULT_CATEGORICAL) + b["n_name_point"])
    z_tmp = art_dir / "z_cf_tmp.npy"
    z_mm = open_memmap(z_tmp, mode="w+", dtype=np.float32, shape=(ub, dim))
    z_off = 0

    rows: list[dict] = []
    chunk = b.get("chunk_apps", 200)
    t0 = time.time()
    for s in range(0, n_apps, chunk):
        sub = apps.iloc[s : s + chunk]
        instances = []
        for _, arow in sub.iterrows():
            rng = np.random.default_rng(seed + int(arow["app_id"]))
            app_is_s1 = arow["app_id"] in s1_apps
            for inst in plan_app(arow, arow["name"], sampler, holdout, app_is_s1, b, rng):
                inst["app_id"] = int(arow["app_id"])
                inst["stage_c_row_idx"] = int(arow["row_idx"])
                inst["subject_amount"] = int(arow["subject_amount"])
                inst["name_cell"] = arow["name_cell"]
                instances.append(inst)

        flat_texts, back = [], []
        for i, inst in enumerate(instances):
            if inst["is_point"]:
                back.append((i, None)); flat_texts.append(inst["cf_text"])
            else:
                for j, t in enumerate(inst["mc_texts"]):
                    back.append((i, j)); flat_texts.append(t)
        amts, _ = decode_amounts(
            model, tokenizer, flat_texts,
            batch_size=b.get("amount_batch_size", 64),
            max_new_tokens=b.get("amount_max_new_tokens", 8), constrained=False,
        )
        point_amt = {}
        mc_amts: dict[int, list] = {}
        for (i, j), a in zip(back, amts):
            if j is None:
                point_amt[i] = a
            else:
                mc_amts.setdefault(i, []).append(a)

        z_idx = [i for i, inst in enumerate(instances)
                 if inst["is_point"] and inst["store_z"] and point_amt.get(i) is not None]
        if z_idx:
            z = collect_anchor_activations(
                model, tokenizer, [instances[i]["cf_text"] for i in z_idx],
                read_layer, batch_size=b.get("z_batch_size", 64), anchor_token_id=anchor_id,
            )
            for k, i in enumerate(z_idx):
                z_mm[z_off] = z[k]; instances[i]["_zrow"] = z_off; z_off += 1

        for i, inst in enumerate(instances):
            sa = inst["subject_amount"]
            base = {k: inst[k] for k in ("app_id", "stage_c_row_idx", "question_class",
                                         "question", "phrase", "split", "split_fine",
                                         "name_cell", "subject_amount")}
            base["target_meta"] = json.dumps(inst["target_meta"])
            if inst["is_point"]:
                a = point_amt.get(i)
                if a is None:
                    continue
                base.update(cf_amount=int(a), cf_amount_std=0.0, mc_n=1,
                            delta=int(a - sa), delta_str=Q.canonical_delta(a - sa),
                            z_cf_row=int(inst.get("_zrow", -1)))
            else:
                vals = [x for x in mc_amts.get(i, []) if x is not None]
                if not vals:
                    continue
                mean = float(np.mean(vals))
                base.update(cf_amount=mean, cf_amount_std=float(np.std(vals)),
                            mc_n=len(vals), delta=mean - sa,
                            delta_str=Q.canonical_delta(mean - sa), z_cf_row=-1)
            rows.append(base)

        done = min(s + chunk, n_apps)
        rate = done / (time.time() - t0)
        print(f"[cfqa] apps {done}/{n_apps} | instances {len(rows)} | "
              f"reruns {len(flat_texts)} last-chunk | {rate:.1f} app/s "
              f"eta {(n_apps-done)/rate/60:.1f}m", flush=True)

    z_mm.flush()
    z_final = art_dir / "z_cf.npy"
    np.save(z_final, np.asarray(z_mm[:z_off]))
    z_tmp.unlink()

    df = pd.DataFrame(rows)
    qa_path = data_dir / "cf_qa.parquet"
    df.to_parquet(qa_path, index=False)

    report = {
        "run_name": run_name, "n_apps": int(n_apps), "n_instances": int(len(df)),
        "model_repo": repo, "adapter": b["adapter"], "read_layer": read_layer,
        "anchor_token_id": anchor_id, "seed": seed,
        "counts_by_class": df["question_class"].value_counts().to_dict(),
        "counts_by_split": df["split"].value_counts().to_dict(),
        "counts_by_split_fine": df["split_fine"].value_counts().to_dict(),
        "n_z_cf": int(z_off), "z_cf_path": str(z_final.relative_to(REPO_ROOT)),
        "z_dim": int(dim),
        "cf_qa_path": str(qa_path.relative_to(REPO_ROOT)),
        "delta_stats": {
            "mean_abs_delta": float(df["delta"].abs().mean()),
            "median_abs_delta": float(df["delta"].abs().median()),
            "frac_zero_delta": float((df["delta"].round(-2) == 0).mean()),
        },
        "name_categorical_spread": {
            "mean_std": float(df.loc[df.question_class == "name_categorical", "cf_amount_std"].mean())
            if (df.question_class == "name_categorical").any() else None,
        },
        "seconds": round(time.time() - t0, 1), "minimal_run": bool(cfg.get("minimal_run")),
    }
    (art_dir / "build_report.json").write_text(json.dumps(report, indent=2, default=str))
    print("[cfqa] report:", json.dumps(report, indent=2, default=str))
    return report
