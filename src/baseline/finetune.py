from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.pcd.finetune import Monitor, _git_commit, _sanitize, load_config
from src.subject.model_io import load_model_and_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def apply_minimal_run(cfg: dict) -> dict:
    t = cfg.setdefault("train", {})
    t["total_steps"] = min(t.get("total_steps") or 8, 8)
    t["per_device_batch_size"] = min(t.get("per_device_batch_size") or 4, 4)
    t["gradient_accumulation_steps"] = 2
    t["log_every_steps"] = 1
    t["val_every_steps"] = 4
    t["checkpoint_step_milestones"] = []
    t["report_to"] = "none"
    cfg.setdefault("qa", {})["val_examples"] = 32
    cfg.setdefault("run", {})
    cfg["run"]["name"] = (cfg["run"].get("name") or "baseline") + "-minimal"
    cfg["minimal_run"] = True
    return cfg


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def build_baseline_example(
    context_ids: list[int],
    question: str,
    delta_str: str,
    tokenizer,
    answer_prompt: str = "\nAnswer:",
    max_len: int = 768,
    eos_id: int | None = None,
) -> tuple[list[int], list[int]]:
    qa_ids = tokenizer(question + answer_prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(" " + delta_str, add_special_tokens=False)["input_ids"]
    if eos_id is not None:
        answer_ids = answer_ids + [eos_id]
    ctx_budget = max_len - len(qa_ids) - len(answer_ids)
    if ctx_budget < 0:
        qa_ids = qa_ids[-max_len + len(answer_ids):]
        ctx_budget = 0
    prompt_ids = context_ids[:ctx_budget] + qa_ids
    return prompt_ids, answer_ids


def collate_baseline(batch: list[dict], pad_id: int) -> dict:
    lens = [len(b["prompt_ids"]) + len(b["answer_ids"]) for b in batch]
    width = max(lens)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        seq = b["prompt_ids"] + b["answer_ids"]
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        labels[i, len(b["prompt_ids"]) : len(seq)] = torch.tensor(b["answer_ids"], dtype=torch.long)
        attn[i, : len(seq)] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": attn}


class BaselineQADataset(Dataset):
    def __init__(self, cfg: dict, tokenizer, split: str, limit: int | None = None,
                 seed: int = 0):
        import pandas as pd

        include_app = cfg.get("include_application", False)
        qa = pd.read_parquet(
            REPO_ROOT / cfg["path"],
            columns=["question", "delta_str", "split", "stage_c_row_idx", "app_id"],
        )
        qa = qa[qa["split"] == split].reset_index(drop=True)
        if len(qa) == 0:
            raise ValueError(f"no QA rows for split {split!r}")
        if limit is not None and len(qa) > limit:
            qa = qa.sample(n=limit, random_state=seed).reset_index(drop=True)

        man = pd.read_parquet(REPO_ROOT / cfg["manifest_path"],
                              columns=["row_idx", "reasoning"]).set_index("row_idx")
        max_len = cfg.get("max_len", 768)
        app_text: dict[int, str] = {}
        if include_app:
            sub = pd.read_parquet(REPO_ROOT / cfg["application_path"],
                                  columns=["app_id", "application_text"])
            app_text = dict(zip(sub["app_id"], sub["application_text"]))

        self.ctx_cache: dict[int, list[int]] = {}
        for ridx, aid in zip(qa["stage_c_row_idx"].unique(),
                             qa.drop_duplicates("stage_c_row_idx")["app_id"]):
            reasoning = man.loc[ridx, "reasoning"]
            text = reasoning
            if include_app and aid in app_text:
                text = reasoning + "\n\n=== APPLICATION ===\n" + app_text[aid]
            self.ctx_cache[int(ridx)] = tokenizer(
                text, add_special_tokens=False, truncation=True, max_length=max_len
            )["input_ids"]

        self.questions = qa["question"].tolist()
        self.deltas = qa["delta_str"].tolist()
        self.rows = qa["stage_c_row_idx"].to_numpy()
        self.tokenizer = tokenizer
        self.answer_prompt = cfg.get("answer_prompt", "\nAnswer:")
        self.max_len = max_len
        self.eos_id = tokenizer.eos_token_id

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, i: int) -> dict:
        prompt_ids, answer_ids = build_baseline_example(
            self.ctx_cache[int(self.rows[i])], self.questions[i], self.deltas[i],
            self.tokenizer, self.answer_prompt, self.max_len, self.eos_id,
        )
        return {"prompt_ids": prompt_ids, "answer_ids": answer_ids}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def load_baseline_model(model_cfg: dict, subject_adapter: str, lora: dict, device):
    from peft import LoraConfig, PeftModel, get_peft_model

    base, tokenizer, repo = load_model_and_tokenizer(model_cfg)
    base = PeftModel.from_pretrained(base, subject_adapter).merge_and_unload()
    peft_cfg = LoraConfig(
        r=int(lora["r"]), lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=list(lora["target_modules"]), task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, peft_cfg).to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, repo


# --------------------------------------------------------------------------
# forward / validation
# --------------------------------------------------------------------------

def qa_forward(model, batch: dict, device) -> dict:
    ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attn = batch["attention_mask"].to(device)
    out = model(input_ids=ids, attention_mask=attn, labels=labels, use_cache=False)
    with torch.no_grad():
        logits = out.logits[:, :-1]
        tgt = labels[:, 1:]
        sup = tgt != -100
        tok_ok = ((logits.argmax(-1) == tgt) & sup).sum()
        seq_ok = (((logits.argmax(-1) == tgt) | ~sup).all(dim=-1)).sum()
    return {"loss": out.loss, "n_answer_tokens": int(sup.sum()),
            "n_tok_correct": int(tok_ok), "n_seq": ids.shape[0],
            "n_seq_correct": int(seq_ok)}


@torch.no_grad()
def run_validation(model, val_dl, accelerator) -> dict:
    model.eval()
    tot = torch.zeros(5, device=accelerator.device)
    for batch in val_dl:
        out = qa_forward(model, batch, accelerator.device)
        tot += torch.tensor(
            [float(out["loss"]) * out["n_answer_tokens"], out["n_answer_tokens"],
             out["n_tok_correct"], out["n_seq"], out["n_seq_correct"]],
            device=accelerator.device)
    tot = accelerator.reduce(tot, reduction="sum")
    model.train()
    return {
        "val_qa_loss": round(float(tot[0] / tot[1]), 4) if tot[1] > 0 else None,
        "val_answer_tok_acc": round(float(tot[2] / tot[1]), 4) if tot[1] > 0 else None,
        "val_exact_match": round(float(tot[4] / tot[3]), 4) if tot[3] > 0 else None,
        "val_n": int(tot[3]),
    }


def save_checkpoint(out_dir: Path, tag: str, model, step: int, cfg: dict, accelerator):
    ckpt_dir = out_dir / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir / "lora")
    (ckpt_dir / "checkpoint_meta.json").write_text(json.dumps({
        "tag": tag, "step": step,
        "include_application": cfg.get("qa", {}).get("include_application", False),
    }, indent=2))
    accelerator.print(f"[baseline] checkpoint {tag} @ step {step}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def run(config_path: str | Path, minimal_run: bool = False) -> dict:
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, set_seed

    cfg = load_config(config_path)
    if minimal_run or cfg.get("minimal_run"):
        cfg = apply_minimal_run(cfg)

    t = cfg["train"]
    qa_cfg = cfg["qa"]
    seed = t.get("seed", 0)

    ddp = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 1),
        kwargs_handlers=[ddp],
    )
    set_seed(seed)
    is_main = accelerator.is_main_process
    rank, world = accelerator.process_index, accelerator.num_processes

    run_name = cfg.get("run", {}).get("name") or "baseline"
    out_dir = REPO_ROOT / t.get("output_root", "artifacts/baselines") / run_name
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    mode = "full-application" if qa_cfg.get("include_application") else "reasoning-only"
    accelerator.print(f"[baseline] {mode} | run dir: {out_dir} | processes: {world}")

    model, tokenizer, repo = load_baseline_model(
        cfg["model"], t["subject_adapter"], t["lora"], accelerator.device)

    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": t.get("weight_decay", 0.01)},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=t.get("learning_rate", 1e-4), betas=tuple(t.get("betas", (0.9, 0.999))))

    total_steps = int(t["total_steps"])
    eff_batch = t["per_device_batch_size"] * t.get("gradient_accumulation_steps", 1) * world

    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=0, num_training_steps=total_steps * world)

    train_ds = BaselineQADataset(qa_cfg, tokenizer,
                                 split=qa_cfg.get("train_split", "pcd_train"), seed=seed)
    pad_id = tokenizer.pad_token_id
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank,
                                 shuffle=True, seed=seed, drop_last=True)
    train_dl = DataLoader(
        train_ds, batch_size=t["per_device_batch_size"], sampler=sampler,
        collate_fn=lambda b: collate_baseline(b, pad_id),
        num_workers=qa_cfg.get("num_workers", 2), pin_memory=True, drop_last=True,
        persistent_workers=qa_cfg.get("num_workers", 2) > 0)
    val_ds = BaselineQADataset(qa_cfg, tokenizer, split=qa_cfg.get("val_split", "sense1_test"),
                               limit=qa_cfg.get("val_examples", 1024), seed=seed)
    val_shard = torch.utils.data.Subset(val_ds, list(range(rank, len(val_ds), world)))
    val_dl = DataLoader(val_shard, batch_size=t["per_device_batch_size"],
                        collate_fn=lambda b: collate_baseline(b, pad_id), num_workers=0)

    model, optim, sched, train_dl = accelerator.prepare(model, optim, sched, train_dl)

    use_wandb = t.get("report_to", "none") == "wandb" and is_main
    if use_wandb:
        import wandb
        wandb.init(project=cfg.get("wandb_project", "pcd-counterfactuals"),
                   name=run_name, config=cfg,
                   mode="offline" if not os.environ.get("WANDB_API_KEY") else "online")

    monitor = Monitor(out_dir, enabled=is_main)
    if is_main:
        (out_dir / "run_manifest.json").write_text(json.dumps({
            "run_name": run_name, "model_repo": repo, "git_commit": _git_commit(),
            "num_processes": world, "effective_batch": eff_batch,
            "total_steps": total_steps, "mode": mode,
            "include_application": qa_cfg.get("include_application", False),
            "train_examples": len(train_ds), "val_examples": len(val_ds),
            "config": cfg,
        }, indent=2, default=str))

    milestones = sorted(int(m) for m in t.get("checkpoint_step_milestones", []))
    ms_idx = 0

    model.train()
    step = 0
    epoch = 0
    ema: float | None = None
    tok_seen, tok_correct = 0, 0
    data_iter = iter(train_dl)
    t_start = time.time()
    accelerator.print(f"[baseline] {total_steps} steps, eff_batch {eff_batch}; "
                      f"{len(train_ds):,} train examples")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(accelerator.device)
    last_val: dict = {}

    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            data_iter = iter(train_dl)
            batch = next(data_iter)
        with accelerator.accumulate(model):
            out = qa_forward(model, batch, accelerator.device)
            loss = out["loss"]
            tok_seen += out["n_answer_tokens"]
            tok_correct += out["n_tok_correct"]
            cur = float(loss.detach().float())
            ema = cur if ema is None else 0.99 * ema + 0.01 * cur
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                if t.get("max_grad_norm"):
                    accelerator.clip_grad_norm_(model.parameters(), t["max_grad_norm"])
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1

                if step % t.get("val_every_steps", 500) == 0 or step == total_steps:
                    last_val = run_validation(model, val_dl, accelerator)
                    accelerator.print(f"[baseline] step {step} val: {last_val}")

                if step % t.get("log_every_steps", 20) == 0 or step == 1:
                    rec = {
                        "qa_loss_ema": round(ema, 4),
                        "train_answer_tok_acc": round(tok_correct / tok_seen, 4) if tok_seen else None,
                        "lr": sched.get_last_lr()[0], "epoch": epoch,
                        "steps_per_sec": round(step / (time.time() - t_start), 3),
                        **last_val,
                    }
                    tok_seen, tok_correct = 0, 0
                    monitor.log(step, total_steps, step * eff_batch, total_steps * eff_batch, rec)
                    if use_wandb:
                        import wandb
                        wandb.log(rec, step=step)
                    accelerator.print(
                        f"[baseline] step {step}/{total_steps} | qa {rec['qa_loss_ema']} | "
                        f"ans_acc {rec['train_answer_tok_acc']} | lr {rec['lr']:.2e}")

                while ms_idx < len(milestones) and step >= milestones[ms_idx]:
                    if is_main:
                        save_checkpoint(out_dir, f"ckpt_step{milestones[ms_idx]}",
                                        accelerator.unwrap_model(model), step, cfg, accelerator)
                    ms_idx += 1

    accelerator.wait_for_everyone()
    elapsed = time.time() - t_start
    if is_main:
        save_checkpoint(out_dir, "final", accelerator.unwrap_model(model), step, cfg, accelerator)
        report = {
            "run_name": run_name, "steps": step, "mode": mode,
            "include_application": qa_cfg.get("include_application", False),
            "elapsed_seconds": round(elapsed, 1),
            "final_qa_loss_ema": _sanitize(ema),
            **{k: _sanitize(v) for k, v in last_val.items()},
            "epochs": epoch, "num_processes": world, "effective_batch": eff_batch,
            "peak_gpu_gib": (round(torch.cuda.max_memory_allocated(accelerator.device) / 2**30, 2)
                             if torch.cuda.is_available() else None),
            "minimal_run": bool(cfg.get("minimal_run")),
        }
        (out_dir / "baseline_report.json").write_text(json.dumps(report, indent=2))
        monitor.status(step, total_steps, step * eff_batch, total_steps * eff_batch, "completed")
        accelerator.print("[baseline] report: " + json.dumps(report, indent=2))
        if use_wandb:
            import wandb
            wandb.finish()
        return report
    return {}
