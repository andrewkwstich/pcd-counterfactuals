from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.subject.decode_eval import decode_amounts, score_predictions

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "artifacts/formula/formula_spec_v1.json"
NAME_SAMPLE_PATH = REPO_ROOT / "data/names/name_sample.parquet"

CELLS = [
    "white_male", "white_female", "asian_male", "asian_female",
    "hispanic_male", "hispanic_female", "black_male", "black_female",
]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def noiseless_amount(df: pd.DataFrame, spec: dict) -> np.ndarray:
    sg = spec["sigmoid"]
    eta0 = df["eta"].to_numpy() - df["eps"].to_numpy()
    g = sg["f_min"] + (sg["f_max"] - sg["f_min"]) * _sigmoid(sg["a"] * eta0 + sg["b"])
    R = df["requested_amount"].to_numpy(dtype=np.float64)
    r100 = spec["round_to"]
    return np.minimum(R, np.round(R * g / r100) * r100)


def invert_g(f_hat: np.ndarray, spec: dict) -> np.ndarray:
    sg = spec["sigmoid"]
    s = np.clip((f_hat - sg["f_min"]) / (sg["f_max"] - sg["f_min"]), 1e-4, 1 - 1e-4)
    return (np.log(s / (1.0 - s)) - sg["b"]) / sg["a"]


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid, float(resid @ resid)


def _f_test(rss0: float, rss1: float, df_extra: int, df_resid: int) -> tuple[float, float]:
    if rss1 <= 0 or df_resid <= 0:
        return float("inf"), 0.0
    F = ((rss0 - rss1) / df_extra) / (rss1 / df_resid)
    return float(F), float(stats.f.sf(F, df_extra, df_resid))


# ---------------------------------------------------------------------------
# formula recovery
# ---------------------------------------------------------------------------

def formula_recovery(df: pd.DataFrame, preds: np.ndarray, spec: dict) -> dict:
    R = df["requested_amount"].to_numpy(dtype=np.float64)
    valid = ~np.isnan(preds)
    uncapped = valid & (preds < 0.98 * R)
    sub = df[uncapped]
    eta_hat = invert_g(preds[uncapped] / R[uncapped], spec)

    q = sub["block_q"].to_numpy()
    c = sub["block_c"].to_numpy()
    nm = sub["block_name"].to_numpy()
    ones = np.ones_like(q)
    n = len(sub)

    X_full = np.stack([ones, q, c, nm], axis=1)
    X_restr = np.stack([ones, q, c], axis=1)
    beta_full, _, rss_full = _ols(X_full, eta_hat)
    _, resid_restr, rss_restr = _ols(X_restr, eta_hat)
    ss_tot = float(((eta_hat - eta_hat.mean()) ** 2).sum())
    F_name, p_name = _f_test(rss_restr, rss_full, 1, n - X_full.shape[1])

    pcs = sub[[f"name_pc{i}" for i in range(1, 7)]].to_numpy()
    X_pc = np.concatenate([ones[:, None], pcs], axis=1)
    beta_pc, _, rss_pc = _ols(X_pc, resid_restr)
    _, _, rss_pc0 = _ols(ones[:, None], resid_restr)
    F_pc, p_pc = _f_test(rss_pc0, rss_pc, pcs.shape[1], n - X_pc.shape[1])

    corr_name = float(np.corrcoef(resid_restr, nm)[0, 1]) if n > 2 else float("nan")
    return {
        "n_uncapped_used": int(n),
        "eta_regression": {
            "coef_intercept": float(beta_full[0]),
            "coef_block_q": float(beta_full[1]),
            "coef_block_c": float(beta_full[2]),
            "coef_block_name": float(beta_full[3]),
            "r2": 1.0 - rss_full / ss_tot if ss_tot > 0 else float("nan"),
            "expected": "coefs ~= 1, intercept ~= eta_center "
                        f"({spec['scales']['eta_center']:.4f})",
        },
        "name_block_partial_F": {"F": F_name, "pvalue": p_name},
        "pc_regression_on_name_residual": {
            "coefs": [float(b) for b in beta_pc[1:]],
            "r2": 1.0 - rss_pc / rss_pc0 if rss_pc0 > 0 else float("nan"),
            "F": F_pc,
            "pvalue": p_pc,
        },
        "corr_residual_vs_block_name": corr_name,
    }


# ---------------------------------------------------------------------------
# per-cell breakdown
# ---------------------------------------------------------------------------

def per_cell_metrics(df: pd.DataFrame, preds: np.ndarray) -> dict:
    out = {}
    for cell in CELLS:
        m = (df["name_cell"] == cell).to_numpy() & ~np.isnan(preds)
        if m.sum() < 2:
            out[cell] = {"n": int(m.sum())}
            continue
        p, a = preds[m], df["amount"].to_numpy(dtype=np.float64)[m]
        ss_tot = float(((a - a.mean()) ** 2).sum())
        out[cell] = {
            "n": int(m.sum()),
            "mae": float(np.abs(p - a).mean()),
            "mean_signed_err": float((p - a).mean()),
            "r2": 1.0 - float(((p - a) ** 2).sum()) / ss_tot if ss_tot > 0 else None,
        }
    return out


# ---------------------------------------------------------------------------
# name-delta reliability
# ---------------------------------------------------------------------------

def name_delta_reliability(
    model,
    tokenizer,
    apps: pd.DataFrame,
    orig_preds: np.ndarray,
    n_names: int,
    batch_size: int,
    seed: int = 0,
    max_new_tokens: int = 8,
) -> dict:
    names_df = pd.read_parquet(NAME_SAMPLE_PATH)
    per_cell = max(1, n_names // len(CELLS))
    parts = [
        g.sample(min(per_cell, len(g)), random_state=seed)
        for _, g in names_df.groupby("cell")
    ]
    probe_names = pd.concat(parts)[["name", "cell"]].reset_index(drop=True)

    texts, keys = [], []
    for m, prow in probe_names.iterrows():
        for k, (_, arow) in enumerate(apps.iterrows()):
            pat = f"Applicant: {arow['name']}\n"
            if arow["application_text"].count(pat) != 1:
                raise ValueError(f"name line not unique for app {arow['app_id']}")
            texts.append(
                arow["application_text"].replace(pat, f"Applicant: {prow['name']}\n", 1)
            )
            keys.append((m, k))

    preds, _ = decode_amounts(
        model, tokenizer, texts, batch_size=batch_size,
        max_new_tokens=max_new_tokens, constrained=True,
    )
    M, K = len(probe_names), len(apps)
    delta = np.full((M, K), np.nan)
    for (m, k), p in zip(keys, preds):
        if p is not None and not np.isnan(orig_preds[k]):
            delta[m, k] = p - orig_preds[k]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(K)
    half_a, half_b = perm[: K // 2], perm[K // 2 :]
    mean_a = np.nanmean(delta[:, half_a], axis=1)
    mean_b = np.nanmean(delta[:, half_b], axis=1)
    ok = ~(np.isnan(mean_a) | np.isnan(mean_b))
    pearson = float(np.corrcoef(mean_a[ok], mean_b[ok])[0, 1]) if ok.sum() > 2 else float("nan")
    spearman = float(stats.spearmanr(mean_a[ok], mean_b[ok]).statistic) if ok.sum() > 2 else float("nan")

    per_name = np.nanmean(delta, axis=1)
    cell_means = {
        cell: float(np.nanmean(per_name[(probe_names["cell"] == cell).to_numpy()]))
        for cell in CELLS
        if (probe_names["cell"] == cell).any()
    }
    return {
        "n_apps": int(K),
        "n_names": int(M),
        "split_half_pearson": pearson,
        "split_half_spearman": spearman,
        "mean_abs_per_name_delta": float(np.nanmean(np.abs(per_name))),
        "per_cell_mean_delta": cell_means,
        "nan_rate": float(np.isnan(delta).mean()),
    }


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------

def run_validation(
    model,
    tokenizer,
    df: pd.DataFrame,
    batch_size: int = 64,
    n_reliability_apps: int = 192,
    n_reliability_names: int = 96,
    thresholds: dict | None = None,
    seed: int = 0,
    max_new_tokens: int = 8,
) -> dict:
    th = {"formula_r2": 0.6, "name_pvalue": 1e-3, "reliability_corr": 0.7,
          **(thresholds or {})}
    spec = json.loads(SPEC_PATH.read_text())
    texts = df["application_text"].tolist()
    amounts = [int(a) for a in df["amount"]]

    free_preds, free_junk = decode_amounts(
        model, tokenizer, texts, batch_size, max_new_tokens, constrained=False)
    cons_preds, cons_junk = decode_amounts(
        model, tokenizer, texts, batch_size, max_new_tokens, constrained=True)
    free = score_predictions(free_preds, amounts, free_junk)
    cons = score_predictions(cons_preds, amounts, cons_junk)

    preds = np.array([np.nan if p is None else float(p) for p in cons_preds])
    a0 = noiseless_amount(df, spec)
    valid = ~np.isnan(preds)
    ss_tot0 = float(((a0[valid] - a0[valid].mean()) ** 2).sum())
    vs_noiseless = {
        "mae": float(np.abs(preds[valid] - a0[valid]).mean()),
        "r2": 1.0 - float(((preds[valid] - a0[valid]) ** 2).sum()) / ss_tot0
              if ss_tot0 > 0 else float("nan"),
    }

    recovery = formula_recovery(df, preds, spec)
    cells = per_cell_metrics(df, preds)

    R = df["requested_amount"].to_numpy(dtype=np.float64)
    eligible = df[valid & (preds < 0.98 * R) & (df["amount"].to_numpy() < R)]
    n_apps = min(n_reliability_apps, len(eligible))
    apps = eligible.sample(n_apps, random_state=seed)
    app_pos = df.index.get_indexer(apps.index)
    reliability = name_delta_reliability(
        model, tokenizer, apps.reset_index(drop=True), preds[app_pos],
        n_reliability_names, batch_size, seed, max_new_tokens,
    )

    gates = {
        "free_decode_clean": bool(free["parse_rate"] >= 0.999
                                  and free["trailing_junk_rate"] <= 0.001),
        "formula_fit": bool(recovery["eta_regression"]["r2"] >= th["formula_r2"]),
        "name_signal": bool(
            recovery["name_block_partial_F"]["pvalue"] < th["name_pvalue"]
            and recovery["pc_regression_on_name_residual"]["pvalue"] < th["name_pvalue"]
        ),
        "name_reliability": bool(
            reliability["split_half_pearson"] >= th["reliability_corr"]),
    }
    return {
        "n_eval": int(len(df)),
        "decode": {"free": free, "constrained": cons},
        "vs_noiseless_formula_mean": vs_noiseless,
        "formula_recovery": recovery,
        "per_cell": cells,
        "name_delta_reliability": reliability,
        "thresholds_provisional": th,
        "gates": gates,
        "b3_pass": bool(all(gates.values())),
    }
