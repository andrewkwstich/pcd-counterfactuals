from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/home-credit")
OUT_DIR = Path("data/applications")
ART_DIR = Path("artifacts/formula")

CORE_FIELDS = [
    "requested_amount",
    "purchase_price",
    "annual_income",
    "monthly_payment",
    "age_years",
]

CLIP_RULES = {
    "annual_income": (0.001, 0.995),
    "requested_amount": (0.001, 0.999),
    "purchase_price": (0.001, 0.999),
    "monthly_payment": (0.001, 0.999),
    "total_outstanding_debt": (0.0, 0.995),
    "avg_payment_delay_days": (0.005, 0.995),
    "avg_cc_utilization": (0.0, 0.995),
}


def load_application() -> pd.DataFrame:
    cols = [
        "SK_ID_CURR",
        "AMT_CREDIT",
        "AMT_GOODS_PRICE",
        "AMT_INCOME_TOTAL",
        "AMT_ANNUITY",
        "DAYS_EMPLOYED",
        "DAYS_BIRTH",
        "CNT_FAM_MEMBERS",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "NAME_INCOME_TYPE",
        "OCCUPATION_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_HOUSING_TYPE",
        "FLAG_OWN_REALTY",
    ]
    df = pd.read_csv(RAW / "application_train.csv", usecols=cols)
    df.loc[df["DAYS_EMPLOYED"] > 0, "DAYS_EMPLOYED"] = np.nan
    out = pd.DataFrame(
        {
            "sk_id_curr": df["SK_ID_CURR"],
            "requested_amount": df["AMT_CREDIT"],
            "purchase_price": df["AMT_GOODS_PRICE"],
            "annual_income": df["AMT_INCOME_TOTAL"],
            "monthly_payment": df["AMT_ANNUITY"],
            "years_employed": (-df["DAYS_EMPLOYED"] / 365.25),
            "age_years": (-df["DAYS_BIRTH"] / 365.25),
            "household_size": df["CNT_FAM_MEMBERS"],
            "ext_score_1": df["EXT_SOURCE_1"],
            "ext_score_2": df["EXT_SOURCE_2"],
            "ext_score_3": df["EXT_SOURCE_3"],
            "employment_type": df["NAME_INCOME_TYPE"],
            "occupation": df["OCCUPATION_TYPE"],
            "education": df["NAME_EDUCATION_TYPE"],
            "housing_type": df["NAME_HOUSING_TYPE"],
            "own_home": (df["FLAG_OWN_REALTY"] == "Y").astype(int),
        }
    )
    return out


def agg_bureau() -> pd.DataFrame:
    df = pd.read_csv(
        RAW / "bureau.csv",
        usecols=["SK_ID_CURR", "CREDIT_ACTIVE", "AMT_CREDIT_SUM_DEBT"],
    )
    active = df[df["CREDIT_ACTIVE"] == "Active"]
    agg = active.groupby("SK_ID_CURR").agg(
        n_open_loans=("CREDIT_ACTIVE", "size"),
        total_outstanding_debt=("AMT_CREDIT_SUM_DEBT", lambda s: s.clip(lower=0).sum()),
    )
    return agg.reset_index().rename(columns={"SK_ID_CURR": "sk_id_curr"})


def agg_prev_apps() -> pd.DataFrame:
    df = pd.read_csv(
        RAW / "previous_application.csv",
        usecols=["SK_ID_CURR", "NAME_CONTRACT_STATUS"],
    )
    decided = df[df["NAME_CONTRACT_STATUS"].isin(["Approved", "Refused"])]
    agg = decided.groupby("SK_ID_CURR")["NAME_CONTRACT_STATUS"].agg(
        prev_approval_rate=lambda s: (s == "Approved").mean(),
        n_prev_decided="size",
    )
    return agg.reset_index().rename(columns={"SK_ID_CURR": "sk_id_curr"})


def agg_installments() -> pd.DataFrame:
    df = pd.read_csv(
        RAW / "installments_payments.csv",
        usecols=["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"],
    )
    df["delay"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
    agg = df.groupby("SK_ID_CURR")["delay"].mean().rename("avg_payment_delay_days")
    return agg.reset_index().rename(columns={"SK_ID_CURR": "sk_id_curr"})


def agg_credit_card() -> pd.DataFrame:
    df = pd.read_csv(
        RAW / "credit_card_balance.csv",
        usecols=["SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"],
    )
    df = df[df["AMT_CREDIT_LIMIT_ACTUAL"] > 0]
    df["util"] = (df["AMT_BALANCE"] / df["AMT_CREDIT_LIMIT_ACTUAL"]).clip(0, 2)
    agg = df.groupby("SK_ID_CURR")["util"].mean().rename("avg_cc_utilization")
    return agg.reset_index().rename(columns={"SK_ID_CURR": "sk_id_curr"})


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    base = load_application()
    n_raw = len(base)

    for agg in (agg_bureau(), agg_prev_apps(), agg_installments(), agg_credit_card()):
        base = base.merge(agg, on="sk_id_curr", how="left")

    base = base.dropna(subset=CORE_FIELDS)
    n_after_core = len(base)

    base["n_open_loans"] = base["n_open_loans"].fillna(0)
    base["total_outstanding_debt"] = base["total_outstanding_debt"].fillna(0)
    base["avg_cc_utilization"] = base["avg_cc_utilization"].fillna(0)
    for col in ["prev_approval_rate", "avg_payment_delay_days", "years_employed",
                "ext_score_1", "ext_score_2", "ext_score_3", "household_size"]:
        base[col] = base[col].fillna(base[col].median())
    base["n_prev_decided"] = base["n_prev_decided"].fillna(0)
    base["occupation"] = base["occupation"].fillna("Other")

    for col, (lo_q, hi_q) in CLIP_RULES.items():
        lo, hi = base[col].quantile(lo_q), base[col].quantile(hi_q)
        base[col] = base[col].clip(lo, hi)

    base = base.reset_index(drop=True)
    base.to_parquet(OUT_DIR / "features_base.parquet", index=False)

    summary = {
        "n_rows_raw": int(n_raw),
        "n_rows_after_core_dropna": int(n_after_core),
        "n_rows_final": int(len(base)),
        "columns": {
            c: {
                "dtype": str(base[c].dtype),
                "n_missing": int(base[c].isna().sum()),
                **(
                    {
                        "mean": float(base[c].mean()),
                        "std": float(base[c].std()),
                        "min": float(base[c].min()),
                        "p50": float(base[c].median()),
                        "max": float(base[c].max()),
                    }
                    if pd.api.types.is_numeric_dtype(base[c])
                    else {"n_unique": int(base[c].nunique())}
                ),
            }
            for c in base.columns
        },
    }
    with open(ART_DIR / "features_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"rows: {n_raw} raw -> {n_after_core} after core dropna -> {len(base)} final")
    print(f"wrote {OUT_DIR / 'features_base.parquet'}")


if __name__ == "__main__":
    build()
