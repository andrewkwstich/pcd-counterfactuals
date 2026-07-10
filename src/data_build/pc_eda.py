from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

OUT = Path("data/names")
ART = Path("artifacts/formula")
N_PCS = 30
TOP_NAMES = 12
RECOVERY_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30]
SEED = 17


def eta_squared(scores: np.ndarray, groups: pd.Series) -> float:
    grand = scores.mean()
    ss_tot = ((scores - grand) ** 2).sum()
    ss_between = sum(
        len(g) * (g.mean() - grand) ** 2
        for _, g in pd.Series(scores).groupby(groups.values)
    )
    return float(ss_between / ss_tot) if ss_tot > 0 else 0.0


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def recovery(scores: np.ndarray, labels: pd.Series, j: int, seed: int = SEED) -> dict:
    X = scores[:, :j]
    clf = LogisticRegression(max_iter=2000, C=1.0)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    acc = cross_val_score(clf, X, labels, cv=cv, scoring="accuracy")
    return {"mean_acc": float(acc.mean()), "std_acc": float(acc.std())}


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    sample = pd.read_parquet(OUT / "name_sample.parquet")
    emb = np.load(OUT / "name_embeddings.npy")
    assert len(sample) == len(emb)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(emb)
    pca = PCA(n_components=N_PCS, random_state=SEED)
    scores = pca.fit_transform(Xs)

    import joblib

    joblib.dump({"scaler": scaler, "pca": pca}, OUT / "pca_model.joblib")
    pc_cols = [f"pc{i+1}" for i in range(N_PCS)]
    pd.concat(
        [sample[["name", "race", "gender", "cell"]].reset_index(drop=True),
         pd.DataFrame(scores, columns=pc_cols)],
        axis=1,
    ).to_parquet(OUT / "name_pc_scores.parquet", index=False)

    races = ["white", "black", "hispanic", "asian"]
    interp = {"explained_variance_ratio": pca.explained_variance_ratio_.tolist(), "pcs": {}}
    is_female = (sample["gender"] == "female").values
    name_len = sample["name"].str.len().values

    for i in range(N_PCS):
        s = scores[:, i]
        order = np.argsort(s)
        entry = {
            "evr": float(pca.explained_variance_ratio_[i]),
            "eta2_race": eta_squared(s, sample["race"]),
            "eta2_gender": eta_squared(s, sample["gender"]),
            "eta2_cell": eta_squared(s, sample["cell"]),
            "d_gender_f_minus_m": cohens_d(s[is_female], s[~is_female]),
            "d_race_one_vs_rest": {
                r: cohens_d(s[(sample["race"] == r).values], s[(sample["race"] != r).values])
                for r in races
            },
            "corr_name_length": float(np.corrcoef(s, name_len)[0, 1]),
            "cell_means": {c: float(s[(sample["cell"] == c).values].mean())
                           for c in sorted(sample["cell"].unique())},
            "low_pole_names": sample["name"].values[order[:TOP_NAMES]].tolist(),
            "high_pole_names": sample["name"].values[order[-TOP_NAMES:]][::-1].tolist(),
        }
        interp["pcs"][f"pc{i+1}"] = entry

    with open(ART / "pc_interpretation.json", "w") as f:
        json.dump(interp, f, indent=2)

    min_cell = sample["cell"].value_counts().min()
    bal_idx = (
        sample.groupby("cell", group_keys=False)
        .apply(lambda g: g.sample(min_cell, random_state=SEED), include_groups=False)
        .index
    )
    bal_mask = sample.index.isin(bal_idx)

    rec = {"balanced_subsample_n": int(bal_mask.sum()), "grid": {}}
    for j in RECOVERY_GRID:
        rec["grid"][str(j)] = {
            "race_full": recovery(scores, sample["race"], j),
            "gender_full": recovery(scores, sample["gender"], j),
            "race_balanced": recovery(scores[bal_mask], sample.loc[bal_mask, "race"], j),
            "gender_balanced": recovery(scores[bal_mask], sample.loc[bal_mask, "gender"], j),
            "cell_balanced": recovery(scores[bal_mask], sample.loc[bal_mask, "cell"], j),
        }
    rec["chance"] = {"race_balanced": 0.25, "gender_balanced": 0.5, "cell_balanced": 0.125}
    with open(ART / "pc_recovery.json", "w") as f:
        json.dump(rec, f, indent=2)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(19, 11))

    ax = axes[0, 0]
    ax.bar(range(1, N_PCS + 1), pca.explained_variance_ratio_ * 100)
    ax.set_title("Variance explained per PC")
    ax.set_xlabel("PC"); ax.set_ylabel("% variance")

    ax = axes[0, 1]
    e_r = [interp["pcs"][f"pc{i+1}"]["eta2_race"] for i in range(N_PCS)]
    e_g = [interp["pcs"][f"pc{i+1}"]["eta2_gender"] for i in range(N_PCS)]
    ax.plot(range(1, N_PCS + 1), e_r, "o-", label="race")
    ax.plot(range(1, N_PCS + 1), e_g, "s-", label="gender")
    ax.set_title(r"$\eta^2$: PC variance explained by label")
    ax.set_xlabel("PC"); ax.set_ylabel(r"$\eta^2$"); ax.legend()

    cmap = {"white_female": "tab:blue", "white_male": "lightsteelblue",
            "black_female": "tab:red", "black_male": "lightcoral",
            "hispanic_female": "tab:green", "hispanic_male": "lightgreen",
            "asian_female": "tab:purple", "asian_male": "plum"}
    for ax, (px, py) in zip(
        [axes[0, 2], axes[1, 0], axes[1, 1]], [(0, 1), (2, 3), (4, 5)]
    ):
        for c, col in cmap.items():
            m = (sample["cell"] == c).values
            ax.scatter(scores[m, px], scores[m, py], s=6, alpha=0.5, c=col, label=c)
        ax.set_xlabel(f"PC{px+1}"); ax.set_ylabel(f"PC{py+1}")
        ax.set_title(f"PC{px+1} vs PC{py+1}")
        if px == 0:
            ax.legend(fontsize=7, markerscale=2)

    ax = axes[1, 2]
    grid = [int(k) for k in rec["grid"]]
    for key, style in [("race_balanced", "o-"), ("gender_balanced", "s-"), ("cell_balanced", "^-")]:
        ax.plot(grid, [rec["grid"][str(j)][key]["mean_acc"] for j in grid], style, label=key)
    ax.axhline(0.25, color="gray", ls=":", lw=1)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.axhline(0.125, color="gray", ls="-.", lw=1)
    ax.set_title("Recovery accuracy vs #PCs (balanced, 5-fold CV)")
    ax.set_xlabel("# leading PCs"); ax.set_ylabel("accuracy"); ax.legend()

    fig.tight_layout()
    fig.savefig(ART / "pc_eda_plots.png", dpi=130)
    print("wrote", ART / "pc_eda_plots.png")

    print("\nEVR (first 10):", np.round(pca.explained_variance_ratio_[:10], 4))
    for i in range(10):
        e = interp["pcs"][f"pc{i+1}"]
        print(f"PC{i+1}: eta2_race={e['eta2_race']:.3f} eta2_gender={e['eta2_gender']:.3f} "
              f"d_gender={e['d_gender_f_minus_m']:+.2f} "
              f"| low: {', '.join(e['low_pole_names'][:5])} | high: {', '.join(e['high_pole_names'][:5])}")


if __name__ == "__main__":
    main()
