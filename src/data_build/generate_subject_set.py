from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_build import formula
from data_build.render import render_application

APP_DIR = Path("data/applications")
NAMES_DIR = Path("data/names")
ART = Path("artifacts/formula")

SEED = 20260706
N_TOTAL = 125_000
SPLIT_SIZES = {"subject_train": 100_000, "subject_val": 5_000, "qa_pool": 20_000}
MAX_DUP = 5
BLEND = 0.5
NAME_HELDOUT_FRAC = 0.10

BALANCE_FIELDS = ["employment_type", "education", "housing_type", "own_home", "occ_bucket"]

OCC_BUCKETS = {
    "Laborers": "labor", "Drivers": "labor", "Low-skill Laborers": "labor",
    "Cleaning staff": "service", "Cooking staff": "service",
    "Security staff": "service", "Waiters/barmen staff": "service",
    "Private service staff": "service",
    "Sales staff": "sales", "Realty agents": "sales",
    "Core staff": "office", "HR staff": "office", "Secretaries": "office",
    "Accountants": "office",
    "High skill tech staff": "tech", "IT staff": "tech",
    "Managers": "mgmt", "Medicine staff": "prof", "Other": "other",
}


def balance_applications(feats: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    feats = feats.copy()
    feats["occ_bucket"] = feats["occupation"].map(lambda x: OCC_BUCKETS.get(x, "other"))

    w = np.ones(len(feats))
    for f in BALANCE_FIELDS:
        emp = feats[f].value_counts(normalize=True)
        uni = 1.0 / len(emp)
        target = {lvl: BLEND * uni + (1 - BLEND) * p for lvl, p in emp.items()}
        w *= feats[f].map(lambda lvl: target[lvl] / emp[lvl]).to_numpy()
    p = w / w.sum()

    counts = np.zeros(len(feats), dtype=int)
    chosen: list[int] = []
    while len(chosen) < N_TOTAL:
        need = N_TOTAL - len(chosen)
        draw = rng.choice(len(feats), size=int(need * 1.2), p=p)
        for i in draw:
            if counts[i] < MAX_DUP:
                counts[i] += 1
                chosen.append(i)
                if len(chosen) == N_TOTAL:
                    break
    out = feats.iloc[chosen].reset_index(drop=True)

    marginals = {
        "seed": SEED, "n_total": N_TOTAL, "blend_uniform": BLEND, "max_dup": MAX_DUP,
        "duplication": {"unique_rows": int((counts > 0).sum()),
                        "max_dup_realized": int(counts.max())},
        "before": {f: feats[f].value_counts(normalize=True).round(4).to_dict()
                   for f in BALANCE_FIELDS},
        "after": {f: out[f].value_counts(normalize=True).round(4).to_dict()
                  for f in BALANCE_FIELDS},
    }
    return out, marginals


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    feats = pd.read_parquet(APP_DIR / "features_base.parquet")
    names = pd.read_parquet(NAMES_DIR / "name_sample.parquet")
    pcs = pd.read_parquet(NAMES_DIR / "name_pc_scores.parquet")
    name_tbl = names[["name", "race", "gender", "cell", "p_female"]].merge(
        pcs[["name"] + formula.PC_COLS], on="name", validate="1:1"
    )

    # ---- balance + name assignment ----
    apps, marginals = balance_applications(feats, rng)
    with open(ART / "balanced_marginals.json", "w") as f:
        json.dump(marginals, f, indent=2)

    idx = rng.integers(0, len(name_tbl), size=len(apps))
    assigned = name_tbl.iloc[idx].reset_index(drop=True)
    apps = pd.concat([apps, assigned.add_prefix("name_")], axis=1)
    apps = apps.rename(columns={"name_name": "name"})

    heldout = set(
        name_tbl.groupby("cell", group_keys=False)["name"]
        .apply(lambda s: s.sample(frac=NAME_HELDOUT_FRAC,
                                  random_state=SEED))
    )
    name_splits = name_tbl[["name", "race", "gender", "cell"]].copy()
    name_splits["name_split"] = np.where(
        name_splits["name"].isin(heldout), "pcd_heldout", "pcd_train"
    )
    name_splits.to_parquet(NAMES_DIR / "name_splits.parquet", index=False)

    # ---- calibrate + freeze ----
    pc_frame = apps[[f"name_{c}" for c in formula.PC_COLS]].copy()
    pc_frame.columns = formula.PC_COLS
    spec = formula.calibrate(apps, pc_frame, apps["name_cell"])
    with open(ART / "formula_spec_v1.json", "w") as f:
        json.dump(spec, f, indent=2)

    # ---- amounts, render, splits ----
    result = formula.eta_and_amount(apps, pc_frame, spec, rng)
    apps = pd.concat([apps, result], axis=1)

    v = {b: float(apps[b].var()) for b in ["block_q", "block_c", "block_name", "eps"]}
    v_eta = float(apps["eta"].var())
    decomp = {
        "var_eta": v_eta,
        "shares_of_var_eta": {k: val / v_eta for k, val in v.items()},
        "cov_q_c_share": float(2 * apps["block_q"].cov(apps["block_c"]) / v_eta),
        "amount_stats": {
            "mean_fraction_of_R": float((apps["amount"] / apps["requested_amount"]).mean()),
            "median_fraction_of_R": float((apps["amount"] / apps["requested_amount"]).median()),
            "p_at_cap": float((apps["amount"] >= apps["requested_amount"]).mean()),
            "p_zero": float((apps["amount"] == 0).mean()),
        },
    }
    with open(ART / "variance_decomposition.json", "w") as f:
        json.dump(decomp, f, indent=2)

    cell_stats = (
        apps.groupby("name_cell")
        .agg(mean_block_name=("block_name", "mean"),
             mean_amount=("amount", "mean"),
             mean_fraction=("approval_fraction", "mean"),
             n=("amount", "size"))
        .round(4)
    )
    cell_stats.to_json(ART / "cell_effects.json", indent=2)

    realized_min = cell_stats["mean_block_name"].idxmin()
    intended_min = min(formula.CELL_TARGETS, key=formula.CELL_TARGETS.get)
    if formula.DISCRIMINATION_DIRECTION == "penalize_white_male":
        intended_min = max(formula.CELL_TARGETS, key=formula.CELL_TARGETS.get)
    if realized_min != intended_min:
        raise RuntimeError(
            f"penalized extreme is {realized_min}, intended {intended_min}; "
            "adjust CELL_TARGETS"
        )

    order = rng.permutation(len(apps))
    tags = np.empty(len(apps), dtype=object)
    start = 0
    for tag, size in SPLIT_SIZES.items():
        tags[order[start:start + size]] = tag
        start += size
    apps["split_tag"] = tags

    apps["application_text"] = [
        render_application(row, row["name"]) for _, row in apps.iterrows()
    ]
    apps["app_id"] = np.arange(len(apps))

    keep = (
        ["app_id", "application_text", "amount", "split_tag", "name",
         "name_race", "name_gender", "name_cell",
         "eta", "eps", "block_q", "block_c", "block_name", "approval_fraction"]
        + [f"name_{c}" for c in formula.PC_COLS]
        + [c for c in feats.columns if c != "sk_id_curr"]
        + ["sk_id_curr"]
    )
    apps[keep].to_parquet(APP_DIR / "subject_set.parquet", index=False)

    print(f"wrote {APP_DIR / 'subject_set.parquet'} ({len(apps)} rows)")
    print("variance shares:", {k: round(x, 4) for k, x in decomp["shares_of_var_eta"].items()})
    print("amount stats:", {k: round(x, 4) for k, x in decomp["amount_stats"].items()})
    print("\nper-cell mean name-block (eta units):")
    print(cell_stats["mean_block_name"].sort_values().to_string())


if __name__ == "__main__":
    main()
