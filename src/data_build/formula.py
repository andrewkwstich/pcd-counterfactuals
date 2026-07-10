from __future__ import annotations

import numpy as np
import pandas as pd

# --- toggles ---------------------------------------------------------------
DISCRIMINATION_DIRECTION = "penalize_black_female"
INTERACTIONS_ENABLED = False

# --- raw (pre-scaling) weights ----------------------------------------------
QUANT_WEIGHTS = {
    "log_income": 0.50,
    "ext_mean": 1.00,
    "pti": -0.60,
    "dti": -0.50,
    "ltv": -0.25,
    "years_employed": 0.30,
    "age_years": 0.15,
    "household_size": -0.15,
    "n_open_loans": -0.20,
    "prev_approval_rate": 0.40,
    "avg_payment_delay_days": -0.40,
    "avg_cc_utilization": -0.35,
}

EDUCATION_OFFSETS = {
    "Lower secondary": -0.60,
    "Secondary / secondary special": 0.00,
    "Incomplete higher": 0.20,
    "Higher education": 0.50,
    "Academic degree": 0.70,
}

EMPLOYMENT_OFFSETS = {
    "Working": 0.00,
    "Commercial associate": 0.10,
    "State servant": 0.20,
    "Pensioner": -0.30,
    "Student": -0.20,
    "Businessman": 0.40,
    "Maternity leave": -0.40,
    "Unemployed": -0.80,
}

OWN_HOME_OFFSET = 0.30

CELL_TARGETS = {
    "white_male": 0.90, "white_female": 0.10,
    "asian_male": 0.30, "asian_female": -0.50,
    "hispanic_male": 0.15, "hispanic_female": -0.65,
    "black_male": -0.10, "black_female": -0.90,
}

PC_COLS = ["pc1", "pc2", "pc3", "pc4", "pc5", "pc6"]

VAR_SHARES = {"quant_cat": 0.65, "name": 0.20, "noise": 0.15}

SIGMOID = {"a": 1.4, "b": 0.955, "f_min": 0.15, "f_max": 1.05}
ROUND_TO = 100


def quant_transforms(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame(index=df.index)
    t["log_income"] = np.log(df["annual_income"])
    t["ext_mean"] = df[["ext_score_1", "ext_score_2", "ext_score_3"]].mean(axis=1)
    t["pti"] = 12 * df["monthly_payment"] / df["annual_income"]
    t["dti"] = df["total_outstanding_debt"] / df["annual_income"]
    t["ltv"] = df["requested_amount"] / df["purchase_price"]
    for c in ["years_employed", "age_years", "household_size", "n_open_loans",
              "prev_approval_rate", "avg_payment_delay_days", "avg_cc_utilization"]:
        t[c] = df[c]
    return t


def derive_pc_weights(pc_z: pd.DataFrame, cells: pd.Series) -> dict:
    sign = -1.0 if DISCRIMINATION_DIRECTION == "penalize_white_male" else 1.0
    centroids = pc_z.groupby(cells.values).mean()
    d = np.array([sign * CELL_TARGETS[c] for c in centroids.index])
    w, *_ = np.linalg.lstsq(centroids.to_numpy(), d, rcond=None)
    return dict(zip(PC_COLS, map(float, w)))


def raw_blocks(df: pd.DataFrame, pc: pd.DataFrame, zstats: dict, pc_weights: dict) -> pd.DataFrame:
    t = quant_transforms(df)
    q = sum(
        w * (t[c] - zstats["quant"][c]["mean"]) / zstats["quant"][c]["std"]
        for c, w in QUANT_WEIGHTS.items()
    )
    c_block = (
        df["education"].map(EDUCATION_OFFSETS).astype(float)
        + df["employment_type"].map(EMPLOYMENT_OFFSETS).astype(float)
        + OWN_HOME_OFFSET * df["own_home"].astype(float)
    )
    n = sum(
        w * (pc[c] - zstats["pc"][c]["mean"]) / zstats["pc"][c]["std"]
        for c, w in pc_weights.items()
    )
    out = pd.DataFrame({"Q": q, "C": c_block, "N": n})
    if out.isna().any().any():
        bad = out.columns[out.isna().any()].tolist()
        raise ValueError(f"NaNs in formula blocks {bad}: unmapped categorical level?")
    return out


def calibrate(df: pd.DataFrame, pc: pd.DataFrame, cells: pd.Series) -> dict:
    t = quant_transforms(df)
    zstats = {
        "quant": {c: {"mean": float(t[c].mean()), "std": float(t[c].std())}
                  for c in QUANT_WEIGHTS},
        "pc": {c: {"mean": float(pc[c].mean()), "std": float(pc[c].std())}
               for c in PC_COLS},
    }
    pc_z = pd.DataFrame(
        {c: (pc[c] - zstats["pc"][c]["mean"]) / zstats["pc"][c]["std"] for c in PC_COLS}
    )
    pc_weights = derive_pc_weights(pc_z, cells)
    blocks = raw_blocks(df, pc, zstats, pc_weights)
    qc = blocks["Q"] + blocks["C"]
    s_qc = float(np.sqrt(VAR_SHARES["quant_cat"] / qc.var()))
    s_n = float(np.sqrt(VAR_SHARES["name"] / blocks["N"].var()))
    sigma_eps = float(np.sqrt(VAR_SHARES["noise"]))
    eta_mean = float(s_qc * qc.mean() + s_n * blocks["N"].mean())
    return {
        "version": "v1",
        "discrimination_direction": DISCRIMINATION_DIRECTION,
        "interactions_enabled": INTERACTIONS_ENABLED,
        "quant_weights": QUANT_WEIGHTS,
        "education_offsets": EDUCATION_OFFSETS,
        "employment_offsets": EMPLOYMENT_OFFSETS,
        "own_home_offset": OWN_HOME_OFFSET,
        "cell_targets": CELL_TARGETS,
        "pc_weights": pc_weights,
        "var_shares_target": VAR_SHARES,
        "sigmoid": SIGMOID,
        "round_to": ROUND_TO,
        "zstats": zstats,
        "scales": {"s_qc": s_qc, "s_n": s_n, "sigma_eps": sigma_eps,
                   "eta_center": eta_mean},
        "placebo_fields": ["occupation", "housing_type"],
    }


def eta_and_amount(
    df: pd.DataFrame, pc: pd.DataFrame, spec: dict, rng: np.random.Generator
) -> pd.DataFrame:
    blocks = raw_blocks(df, pc, spec["zstats"], spec["pc_weights"])
    s = spec["scales"]
    eps = rng.normal(0.0, s["sigma_eps"], size=len(df))
    eta = (s["s_qc"] * (blocks["Q"] + blocks["C"])
           + s["s_n"] * blocks["N"] - s["eta_center"] + eps)
    sg = spec["sigmoid"]
    g = sg["f_min"] + (sg["f_max"] - sg["f_min"]) / (1.0 + np.exp(-(sg["a"] * eta + sg["b"])))
    r = df["requested_amount"].to_numpy(dtype=float)
    amount = np.minimum(r, np.round(r * g / spec["round_to"]) * spec["round_to"])
    return pd.DataFrame(
        {
            "eta": eta,
            "eps": eps,
            "block_q": s["s_qc"] * blocks["Q"].to_numpy(),
            "block_c": s["s_qc"] * blocks["C"].to_numpy(),
            "block_name": s["s_n"] * blocks["N"].to_numpy(),
            "approval_fraction": g,
            "amount": amount.astype(int),
        },
        index=df.index,
    )
