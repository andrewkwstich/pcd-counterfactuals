from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def collect_anchor_activations(
    model,
    tokenizer,
    texts: list[str],
    read_layer: int,
    batch_size: int = 32,
    anchor_token_id: int = 400,
) -> np.ndarray:
    device = next(model.parameters()).device
    bos = tokenizer.bos_token_id
    pad = tokenizer.pad_token_id
    feats: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        encs = tokenizer(chunk, add_special_tokens=False)["input_ids"]
        seqs = [[bos] + e for e in encs]
        max_len = max(len(s) for s in seqs)
        input_ids = torch.tensor(
            [s + [pad] * (max_len - len(s)) for s in seqs], device=device
        )
        attention = torch.tensor(
            [[1] * len(s) + [0] * (max_len - len(s)) for s in seqs], device=device
        )
        out = model(input_ids=input_ids, attention_mask=attention,
                    output_hidden_states=True)
        h = out.hidden_states[read_layer]
        for j, s in enumerate(seqs):
            pos = len(s) - 1
            if s[pos] != anchor_token_id:
                raise ValueError(f"prompt {i + j} does not end at anchor token")
            feats.append(h[j, pos].float().cpu().numpy())
    return np.stack(feats)


def fit_ridge_probe(
    X: np.ndarray, y: np.ndarray, test_frac: float = 0.2, seed: int = 0
) -> dict:
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_frac, random_state=seed
    )
    probe = make_pipeline(
        StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 15))
    )
    probe.fit(X_tr, y_tr)
    return {
        "r2_train": float(probe.score(X_tr, y_tr)),
        "r2_test": float(probe.score(X_te, y_te)),
        "alpha": float(probe[-1].alpha_),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
    }


def run_probe(
    model,
    tokenizer,
    df,
    read_layer: int,
    anchor_token_id: int,
    batch_size: int = 32,
    test_frac: float = 0.2,
    threshold_r2: float = 0.8,
    seed: int = 0,
) -> dict:
    texts = df["application_text"].tolist()
    X = collect_anchor_activations(
        model, tokenizer, texts, read_layer, batch_size, anchor_token_id
    )
    amount = df["amount"].to_numpy(dtype=np.float64)
    targets = {
        "amount": amount,
        "log_amount": np.log(amount),
        "approval_fraction": df["approval_fraction"].to_numpy(dtype=np.float64),
    }
    report: dict = {
        "read_layer": read_layer,
        "hidden_state_convention": "hidden_states[k] = residual stream after block k",
        "anchor_token_id": anchor_token_id,
        "n_examples": int(len(df)),
        "d_model": int(X.shape[1]),
        "probes": {},
    }
    for name, y in targets.items():
        report["probes"][name] = fit_ridge_probe(X, y, test_frac, seed)
    R = df["requested_amount"].to_numpy(dtype=np.float64)
    baseline_X = np.stack([R, np.log(R)], axis=1)
    report["baseline_requested_only"] = fit_ridge_probe(
        baseline_X, amount, test_frac, seed
    )
    r2 = report["probes"]["amount"]["r2_test"]
    report["threshold_r2"] = threshold_r2
    report["gate_pass"] = bool(r2 >= threshold_r2)
    return report
