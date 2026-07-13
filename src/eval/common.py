from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

from src.baseline.finetune import build_baseline_example
from src.pcd.finetune import build_qa_example, load_frozen_encoder
from src.subject.model_io import load_model_and_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]

SLICES = ["sense1_test", "sense2_names", "sense2_race", "sense2_phrasing", "sense2_attribute"]
GEN_SLICES = ["sense1_test", "sense2_names", "sense2_race", "sense2_phrasing"]
CLASSES = ["categorical", "name_categorical", "name_point", "quant"]

KINDS = {
    "pcd": {"run": "artifacts/pcd/pcd-finetune-v1"},
    "f1": {"run": "artifacts/baselines/baseline-f1-v1"},
    "f1prime": {"run": "artifacts/baselines/baseline-f1prime-v1"},
    "f2": {"run": "artifacts/baselines/baseline-f2-v1"},
    "f1redact": {
        "run": "artifacts/baselines/baseline-f1-v1",
        "reasoning": "artifacts/activations/qa_pool/reasoning_noname.parquet",
        "zero_baseline": False,
    },
}

_DELTA_RE = re.compile(r"\s*([+-]?[0-9][0-9,]*)")


def parse_delta(text: str) -> int | None:
    m = _DELTA_RE.match(text)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return None


def load_run_config(run: str) -> dict:
    manifest = json.loads((REPO_ROOT / run / "run_manifest.json").read_text())
    return manifest["config"]


def load_qa(cfg: dict):
    import pandas as pd

    return pd.read_parquet(
        REPO_ROOT / cfg["qa"]["path"],
        columns=["question", "delta", "delta_str", "split_fine", "question_class",
                 "stage_c_row_idx", "app_id", "cf_amount_std", "mc_n"],
    )


def load_eval_qa(cfg: dict):
    df = load_qa(cfg)
    return df[df.delta != 0].reset_index(drop=True)


def eval_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scramble_perm(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n)


def label_sign_ceiling(cell) -> float:
    from math import erf, sqrt

    se = cell["cf_amount_std"].to_numpy() / np.sqrt(cell["mc_n"].to_numpy())
    d = np.abs(cell["delta"].to_numpy())
    z = np.divide(d, se, out=np.full(len(cell), np.inf), where=se > 0)
    p = np.array([0.5 * (1.0 + erf(x / sqrt(2))) for x in z])
    return float(p.mean())


def sign_token_ids(tokenizer) -> tuple[int, int]:
    plus = tokenizer(" +", add_special_tokens=False)["input_ids"]
    minus = tokenizer(" -", add_special_tokens=False)["input_ids"]
    return plus[0], minus[0]


class PCDEvalModel:
    def __init__(self, run: str, device):
        from peft import PeftModel

        cfg = load_run_config(run)
        base, tokenizer, _ = load_model_and_tokenizer(cfg["model"])
        base = PeftModel.from_pretrained(
            base, str(REPO_ROOT / cfg["train"]["subject_adapter"])
        ).merge_and_unload()
        decoder = PeftModel.from_pretrained(base, str(REPO_ROOT / run / "final/decoder_lora"))
        decoder.to(device).eval().requires_grad_(False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        encoder = load_frozen_encoder(cfg["encoder"], device)

        self.cfg = cfg
        self.device = device
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.embed = decoder.get_input_embeddings()
        self.dtype = next(decoder.parameters()).dtype
        self.answer_prompt = cfg["qa"].get("answer_prompt", "\nAnswer:")
        self.z = np.load(REPO_ROOT / cfg["qa"]["z_path"], mmap_mode="r")

    def z_batch(self, rows: np.ndarray, variant: str, perm: np.ndarray | None) -> torch.Tensor:
        if variant == "zeros":
            z = np.zeros((len(rows), self.z.shape[1]), dtype=np.float32)
        else:
            src = perm[rows] if variant == "scramble" else rows
            z = np.array(self.z[src], dtype=np.float32)
        return torch.from_numpy(z).to(self.device)

    def prompt_ids(self, question: str, delta_str: str) -> tuple[list[int], list[int]]:
        return build_qa_example(question, delta_str, self.tokenizer,
                                self.answer_prompt, self.tokenizer.eos_token_id)


class TextEvalModel:
    def __init__(self, run: str, device, reasoning_path: str | None = None):
        import pandas as pd
        from peft import PeftModel

        cfg = load_run_config(run)
        base, tokenizer, _ = load_model_and_tokenizer(cfg["model"])
        base = PeftModel.from_pretrained(
            base, str(REPO_ROOT / cfg["train"]["subject_adapter"])
        ).merge_and_unload()
        model = PeftModel.from_pretrained(base, str(REPO_ROOT / run / "final/lora"))
        model.to(device).eval().requires_grad_(False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        qa_cfg = cfg["qa"]
        man = pd.read_parquet(
            REPO_ROOT / (reasoning_path or qa_cfg["manifest_path"]),
            columns=["row_idx", "app_id", "reasoning"],
        )
        app_text: dict[int, str] = {}
        if qa_cfg.get("include_application"):
            sub = pd.read_parquet(REPO_ROOT / qa_cfg["application_path"],
                                  columns=["app_id", "application_text"])
            app_text = dict(zip(sub["app_id"], sub["application_text"]))

        max_len = qa_cfg.get("max_len", 768)
        self.ctx_cache: dict[int, list[int]] = {}
        for ridx, aid, reasoning in zip(man["row_idx"], man["app_id"], man["reasoning"]):
            text = reasoning
            if app_text and aid in app_text:
                text = reasoning + "\n\n=== APPLICATION ===\n" + app_text[aid]
            self.ctx_cache[int(ridx)] = tokenizer(
                text, add_special_tokens=False, truncation=True, max_length=max_len
            )["input_ids"]

        self.cfg = cfg
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.answer_prompt = qa_cfg.get("answer_prompt", "\nAnswer:")
        self.max_len = max_len

    def context_ids(self, row: int, variant: str, perm: np.ndarray | None) -> list[int]:
        if variant == "qonly":
            return []
        src = int(perm[row]) if variant == "scramble" else int(row)
        return self.ctx_cache[src]

    def prompt_ids(self, context_ids: list[int], question: str,
                   delta_str: str) -> tuple[list[int], list[int]]:
        return build_baseline_example(context_ids, question, delta_str, self.tokenizer,
                                      self.answer_prompt, self.max_len,
                                      self.tokenizer.eos_token_id)


@torch.no_grad()
def fc_margins_text(m: TextEvalModel, qa, variant: str, perm: np.ndarray | None,
                    batch_size: int) -> np.ndarray:
    plus_id, minus_id = sign_token_ids(m.tokenizer)
    pad_id = m.tokenizer.pad_token_id
    prompts = [
        m.prompt_ids(m.context_ids(row, variant, perm), q, d)[0]
        for row, q, d in zip(qa["stage_c_row_idx"], qa["question"], qa["delta_str"])
    ]
    margins = np.empty(len(prompts), dtype=np.float32)
    for lo in range(0, len(prompts), batch_size):
        chunk = prompts[lo : lo + batch_size]
        lens = [len(p) for p in chunk]
        width = max(lens)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), width), dtype=torch.long)
        for i, p in enumerate(chunk):
            ids[i, : len(p)] = torch.tensor(p, dtype=torch.long)
            attn[i, : len(p)] = 1
        logits = m.model(input_ids=ids.to(m.device), attention_mask=attn.to(m.device),
                         use_cache=False).logits
        pos = torch.tensor(lens, device=m.device) - 1
        last = logits[torch.arange(len(chunk), device=m.device), pos]
        margins[lo : lo + len(chunk)] = (
            (last[:, plus_id] - last[:, minus_id]).float().cpu().numpy()
        )
    return margins


@torch.no_grad()
def fc_margins_pcd(m: PCDEvalModel, qa, variant: str, perm: np.ndarray | None,
                   batch_size: int) -> np.ndarray:
    plus_id, minus_id = sign_token_ids(m.tokenizer)
    pad_id = m.tokenizer.pad_token_id
    prompts = [m.prompt_ids(q, d)[0] for q, d in zip(qa["question"], qa["delta_str"])]
    rows = qa["stage_c_row_idx"].to_numpy()
    margins = np.empty(len(prompts), dtype=np.float32)
    for lo in range(0, len(prompts), batch_size):
        chunk = prompts[lo : lo + batch_size]
        lens = [len(p) for p in chunk]
        width = max(lens)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), width + 1), dtype=torch.long)
        for i, p in enumerate(chunk):
            ids[i, : len(p)] = torch.tensor(p, dtype=torch.long)
            attn[i, : len(p) + 1] = 1
        z = m.z_batch(rows[lo : lo + len(chunk)], variant, perm)
        enc = m.encoder.encode(z.unsqueeze(1), out_dtype=m.dtype)
        embeds = m.embed(ids.to(m.device))
        inputs_embeds = torch.cat([enc.soft_tokens, embeds], dim=1)
        logits = m.decoder(inputs_embeds=inputs_embeds, attention_mask=attn.to(m.device),
                           use_cache=False).logits
        pos = torch.tensor(lens, device=m.device)
        last = logits[torch.arange(len(chunk), device=m.device), pos]
        margins[lo : lo + len(chunk)] = (
            (last[:, plus_id] - last[:, minus_id]).float().cpu().numpy()
        )
    return margins


def fc_correct(margins: np.ndarray, deltas) -> np.ndarray:
    return (margins > 0) == (np.asarray(deltas) > 0)


@torch.no_grad()
def generate_text(m: TextEvalModel, qa, variant: str, perm: np.ndarray | None,
                  batch_size: int, max_new_tokens: int = 8) -> list[int | None]:
    pad_id = m.tokenizer.pad_token_id
    eos_id = m.tokenizer.eos_token_id
    prompts = [
        m.prompt_ids(m.context_ids(row, variant, perm), q, d)[0]
        for row, q, d in zip(qa["stage_c_row_idx"], qa["question"], qa["delta_str"])
    ]
    preds: list[int | None] = []
    for lo in range(0, len(prompts), batch_size):
        chunk = prompts[lo : lo + batch_size]
        width = max(len(p) for p in chunk)
        ids = torch.tensor([[pad_id] * (width - len(p)) + p for p in chunk],
                           device=m.device)
        attn = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in chunk],
                            device=m.device)
        out = m.model.generate(
            input_ids=ids, attention_mask=attn, max_new_tokens=max_new_tokens,
            do_sample=False, eos_token_id=eos_id, pad_token_id=pad_id, use_cache=True,
        )
        for row in out[:, width:]:
            text = m.tokenizer.decode(row, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)
            preds.append(parse_delta(text))
    return preds


@torch.no_grad()
def generate_pcd(m: PCDEvalModel, qa, variant: str, perm: np.ndarray | None,
                 batch_size: int, max_new_tokens: int = 8) -> list[int | None]:
    pad_id = m.tokenizer.pad_token_id
    eos_id = m.tokenizer.eos_token_id
    prompts = [m.prompt_ids(q, d)[0] for q, d in zip(qa["question"], qa["delta_str"])]
    rows = qa["stage_c_row_idx"].to_numpy()
    preds: list[int | None] = []
    for lo in range(0, len(prompts), batch_size):
        chunk = prompts[lo : lo + batch_size]
        width = max(len(p) for p in chunk) + 1
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), width), dtype=torch.long)
        for i, p in enumerate(chunk):
            ids[i, width - len(p):] = torch.tensor(p, dtype=torch.long)
            attn[i, width - len(p) - 1:] = 1
        z = m.z_batch(rows[lo : lo + len(chunk)], variant, perm)
        enc = m.encoder.encode(z.unsqueeze(1), out_dtype=m.dtype)
        embeds = m.embed(ids.to(m.device))
        for i, p in enumerate(chunk):
            embeds[i, width - len(p) - 1] = enc.soft_tokens[i, 0]
        out = m.decoder.generate(
            inputs_embeds=embeds, attention_mask=attn.to(m.device),
            max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=eos_id, pad_token_id=pad_id, use_cache=True,
        )
        for row in out:
            text = m.tokenizer.decode(row, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)
            preds.append(parse_delta(text))
    return preds


def gen_metrics(preds: list[int | None], deltas) -> dict:
    d = np.asarray(deltas, dtype=float)
    pred = np.array([p if p is not None else 0 for p in preds], dtype=float)
    correct = ((pred > 0) & (d > 0)) | ((pred < 0) & (d < 0))
    committed = pred != 0
    return {
        "n": len(preds),
        "parse_rate": sum(p is not None for p in preds) / len(preds),
        "sign_acc": float(correct.mean()),
        "sign_acc_committed": float(correct[committed].mean()) if committed.any() else None,
        "pred_zero_rate": float((~committed).mean()),
        "mae": float(np.abs(pred - d).mean()),
        "zero_baseline_mae": float(np.abs(d).mean()),
    }


def sample_cell(cell, n: int, seed: int):
    if len(cell) <= n:
        return cell
    return cell.sample(n=n, random_state=seed)


def class_sample(qa, per_class: int, seed: int):
    import pandas as pd

    s1 = qa[qa.split_fine == "sense1_test"]
    return pd.concat(
        [s1[s1.question_class == c].sample(n=per_class, random_state=seed) for c in CLASSES]
    )
