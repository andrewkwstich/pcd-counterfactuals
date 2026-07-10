from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/names_src")
SSA_RDA = Path("data/raw/names_src/babynames.rda")
OUT = Path("data/names")
FASTTEXT_MODEL = Path("data/raw/cc.en.300.bin")

RACES = {"whi": "white", "bla": "black", "his": "hispanic", "asi": "asian"}

PROB_RENAMES = {
    **{f"{c}_rgn": f"p_{r}_given_name" for c, r in {**RACES, 'oth': 'other'}.items()},
    **{f"{c}_ngr": f"p_name_given_{r}" for c, r in {**RACES, 'oth': 'other'}.items()},
}

# --- thresholds ------------------------------------------------------------
RACE_PURITY_MIN = 0.70
GENDER_CONF_MIN = 0.90
SSA_MIN_BIRTHS = 200
CELL_SIZE = 500
SSA_YEAR_MIN = 1950


def load_race_dicts() -> tuple[pd.DataFrame, pd.DataFrame]:
    p_race_given_name = pd.read_csv(RAW / "first_nameRaceProbs.csv")
    p_name_given_race = pd.read_csv(RAW / "first_raceNameProbs.csv")
    for df in (p_race_given_name, p_name_given_race):
        df["name"] = df["name"].astype(str).str.strip().str.title()
    return p_race_given_name, p_name_given_race


def load_ssa_gender() -> pd.DataFrame:
    import pyreadr

    allb = pyreadr.read_r(str(SSA_RDA))["babynames"]
    allb = allb[allb["year"] >= SSA_YEAR_MIN]
    g = allb.pivot_table(index="name", columns="sex", values="n", aggfunc="sum").fillna(0)
    g["total_births"] = g["F"] + g["M"]
    g["p_female"] = g["F"] / g["total_births"]
    g = g.reset_index()[["name", "total_births", "p_female"]]
    g["name"] = g["name"].astype(str).str.title()
    return g


def add_names_dataset_gender(pool: pd.DataFrame) -> pd.DataFrame:
    from names_dataset import NameDataset

    nd = NameDataset()
    ssa_ok = pool["p_female"].notna() & (pool["total_births"] >= SSA_MIN_BIRTHS)
    pool["gender_source"] = np.where(ssa_ok, "ssa", None)
    pool["top_country"] = None

    p_f_nd, top_c = {}, {}
    for n in pool["name"]:
        r = nd.search(n).get("first_name")
        if not r:
            continue
        g = r.get("gender") or {}
        if g.get("Female") is not None:
            p_f_nd[n] = float(g["Female"])
        c = {k: v for k, v in (r.get("country") or {}).items() if v}
        if c:
            top_c[n] = max(c, key=c.get)

    pool["top_country"] = pool["name"].map(top_c)
    nd_fill = ~ssa_ok & pool["name"].isin(p_f_nd)
    pool.loc[nd_fill, "p_female"] = pool.loc[nd_fill, "name"].map(p_f_nd)
    pool.loc[nd_fill, "gender_source"] = "names_dataset"
    return pool


def build_pool() -> pd.DataFrame:
    p_rgn, p_ngr = load_race_dicts()
    ssa = load_ssa_gender()

    pool = p_rgn.merge(
        p_ngr, on="name", suffixes=("_rgn", "_ngr"), how="inner"
    ).merge(ssa, on="name", how="left")

    ok = pool["name"].str.fullmatch(r"[A-Za-z][A-Za-z'\-]{1,14}")
    pool = pool[ok].copy()
    pool = add_names_dataset_gender(pool)
    return pool


def balanced_sample(pool: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    covered = pool[pool["gender_source"].notna()].copy()

    rows = []
    report_cells = {}
    for rcode, rname in RACES.items():
        purity = covered[covered[f"{rcode}_rgn"] >= RACE_PURITY_MIN]
        for gender in ("female", "male"):
            conf = (
                purity[purity["p_female"] >= GENDER_CONF_MIN]
                if gender == "female"
                else purity[purity["p_female"] <= 1 - GENDER_CONF_MIN]
            )
            ranked = conf.sort_values(f"{rcode}_ngr", ascending=False)
            take = ranked.head(CELL_SIZE).copy()
            take["race"] = rname
            take["gender"] = gender
            take["cell"] = f"{rname}_{gender}"
            rows.append(take)
            report_cells[f"{rname}_{gender}"] = {
                "eligible": int(len(conf)),
                "taken": int(len(take)),
                "min_p_name_given_race_taken": float(take[f"{rcode}_ngr"].min()) if len(take) else None,
                "gender_source_counts": take["gender_source"].value_counts().to_dict(),
            }

    sample = pd.concat(rows, ignore_index=True)
    race_prob_cols = {"white": "whi_rgn", "black": "bla_rgn", "hispanic": "his_rgn", "asian": "asi_rgn"}
    sample["own_cell_purity"] = sample.apply(lambda r: r[race_prob_cols[r["race"]]], axis=1)
    sample = (
        sample.sort_values("own_cell_purity", ascending=False)
        .drop_duplicates(subset="name", keep="first")
        .reset_index(drop=True)
    )
    sample = sample.rename(columns=PROB_RENAMES)
    report = {
        "thresholds": {
            "race_purity_min": RACE_PURITY_MIN,
            "gender_conf_min": GENDER_CONF_MIN,
            "ssa_min_births": SSA_MIN_BIRTHS,
            "cell_size_target": CELL_SIZE,
            "ssa_year_min": SSA_YEAR_MIN,
        },
        "pool_size": int(len(pool)),
        "pool_gender_coverage_by_source": pool["gender_source"].value_counts(dropna=False).to_dict(),
        "cells": report_cells,
        "final_counts": sample["cell"].value_counts().to_dict(),
        "sample_gender_source_counts": sample["gender_source"].value_counts().to_dict(),
    }
    return sample, report


def embed(sample: pd.DataFrame) -> np.ndarray:
    import fasttext

    model = fasttext.load_model(str(FASTTEXT_MODEL))
    vecs = np.stack([model.get_word_vector(n) for n in sample["name"]])
    return vecs.astype(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = build_pool()
    pool.rename(columns=PROB_RENAMES).to_parquet(OUT / "name_pool.parquet", index=False)

    sample, report = balanced_sample(pool)
    sample.to_parquet(OUT / "name_sample.parquet", index=False)
    with open(OUT / "sampling_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("cell counts:\n", sample["cell"].value_counts().to_string())

    vecs = embed(sample)
    np.save(OUT / "name_embeddings.npy", vecs)
    print(f"embeddings: {vecs.shape} -> {OUT / 'name_embeddings.npy'}")


if __name__ == "__main__":
    main()
