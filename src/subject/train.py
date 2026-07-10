from __future__ import annotations

import json
import math
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.subject.data import CompletionCollator, SubjectDataset, load_split
from src.subject.decode_eval import numeric_decode_eval
from src.subject.model_io import load_model_and_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# config handling
# --------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_minimal_run(cfg: dict) -> dict:
    d, t = cfg["data"], cfg["train"]
    d["max_train_examples"] = min(d.get("max_train_examples") or 100, 100)
    d["max_val_examples"] = min(d.get("max_val_examples") or 50, 50)
    t["max_steps"] = min(t.get("max_steps") or 10, 10)
    t["num_epochs"] = 1
    t["eval_steps"] = 5
    t["logging_steps"] = 1
    t["save_strategy"] = "no"
    t["decode_eval_examples"] = min(t.get("decode_eval_examples") or 16, 16)
    cfg.setdefault("run", {})
    cfg["run"]["name"] = (cfg["run"].get("name") or "subject") + "-minimal"
    cfg["minimal_run"] = True
    return cfg


# --------------------------------------------------------------------------
# callbacks
# --------------------------------------------------------------------------

def _sanitize(v: Any) -> Any:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


class HeartbeatCallback(TrainerCallback):
    def __init__(self, run_dir: Path) -> None:
        self.metrics_path = run_dir / "metrics.jsonl"
        self.status_path = run_dir / "status.json"
        self.t0: float | None = None
        self.last: dict[str, Any] = {}

    def _write_status(self, state, status: str, extra: dict | None = None) -> None:
        max_steps = getattr(state, "max_steps", 0) or 0
        step = getattr(state, "global_step", 0) or 0
        pct = 100.0 * step / max_steps if max_steps else None
        eta = None
        if self.t0 and step and max_steps and status == "running":
            eta = (time.time() - self.t0) / step * (max_steps - step)
        payload = {
            "state": status,
            "global_step": step,
            "max_steps": max_steps,
            "pct_complete": round(pct, 2) if pct is not None else None,
            "epoch": round(state.epoch, 4) if state and state.epoch else None,
            "eta_seconds": round(eta) if eta is not None else None,
            "elapsed_seconds": round(time.time() - self.t0) if self.t0 else None,
            "pid": os.getpid(),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **self.last,
            **(extra or {}),
        }
        self.status_path.write_text(json.dumps(payload, indent=2))

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self.t0 = time.time()
        self._write_status(state, "running")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "step": state.global_step,
            "max_steps": state.max_steps,
            "epoch": round(state.epoch, 4) if state.epoch else None,
            **{k: _sanitize(v) for k, v in logs.items()},
        }
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if "loss" in logs:
            self.last["last_train_loss"] = _sanitize(logs["loss"])
        if "eval_loss" in logs:
            self.last["last_eval_loss"] = _sanitize(logs["eval_loss"])
        if "decode/mae" in logs:
            self.last["last_decode_mae"] = _sanitize(logs["decode/mae"])
            self.last["last_decode_parse_rate"] = _sanitize(logs.get("decode/parse_rate"))
        self._write_status(state, "running")

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._write_status(state, "completed")

    def mark_failed(self, err: str) -> None:
        base = {}
        if self.status_path.exists():
            base = json.loads(self.status_path.read_text())
        base.update(state="failed", error=err,
                    updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        self.status_path.write_text(json.dumps(base, indent=2))


class DecodeEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, texts, amounts, batch_size, max_new_tokens,
                 constrained: bool = False):
        self.tokenizer = tokenizer
        self.texts = texts
        self.amounts = amounts
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.constrained = constrained
        self.trainer: Trainer | None = None
        self.last_metrics: dict = {}

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero or model is None or not self.texts:
            return
        t0 = time.time()
        m = numeric_decode_eval(
            model, self.tokenizer, self.texts, self.amounts,
            batch_size=self.batch_size, max_new_tokens=self.max_new_tokens,
            constrained=self.constrained,
        )
        m["seconds"] = round(time.time() - t0, 1)
        self.last_metrics = m
        logs = {f"decode/{k}": v for k, v in m.items()}
        if self.trainer is not None:
            self.trainer.log(logs)
        else:
            print(f"[decode_eval] {logs}")


# --------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------

def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def _configure_wandb(report_to: str, run_name: str) -> str:
    if "wandb" not in report_to:
        return report_to
    os.environ.setdefault("WANDB_PROJECT", "pcd-counterfactuals")
    if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") != "offline":
        print("[train_subject] WANDB_API_KEY unset -> WANDB_MODE=offline "
              "(sync later with `wandb sync`)")
        os.environ["WANDB_MODE"] = "offline"
    return report_to


def run(config_path: str | Path, minimal_run: bool = False) -> dict:
    cfg = load_config(config_path)
    if minimal_run or cfg.get("minimal_run"):
        cfg = apply_minimal_run(cfg)
    d, t = cfg["data"], cfg["train"]
    run_cfg = cfg.setdefault("run", {})
    run_name = run_cfg.get("name") or "subject-lora"
    output_root = REPO_ROOT / run_cfg.get("output_root", "artifacts/subject")
    run_dir = output_root / run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not cfg.get("minimal_run"):
        run_dir = output_root / f"{run_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train_subject] run dir: {run_dir}")

    torch.manual_seed(t.get("seed", 17))

    # ---- data ------------------------------------------------------------
    data_path = str(REPO_ROOT / d["path"])
    train_df = load_split(data_path, d["train_split"], d.get("max_train_examples"))
    val_df = load_split(data_path, d["val_split"], d.get("max_val_examples"))

    # ---- model -----------------------------------------------------------
    model, tokenizer, model_repo = load_model_and_tokenizer(cfg["model"])

    anchor_id = cfg["anchor"]["token_id"]
    train_ds = SubjectDataset(train_df, tokenizer,
                              d.get("text_col", "application_text"),
                              d.get("label_col", "amount"), anchor_id)
    val_ds = SubjectDataset(val_df, tokenizer,
                            d.get("text_col", "application_text"),
                            d.get("label_col", "amount"), anchor_id)

    max_seq_len = t.get("max_seq_len", 256)
    longest = max(len(ex["input_ids"]) for ex in train_ds.examples + val_ds.examples)
    if longest > max_seq_len:
        raise ValueError(
            f"longest example is {longest} tokens > max_seq_len {max_seq_len}; "
            "truncation would destroy the trailing anchor — fix data or raise limit"
        )

    lora_cfg = t["lora"]
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    if t.get("gradient_checkpointing"):
        model.enable_input_require_grads()

    # ---- trainer ---------------------------------------------------------
    max_steps = t.get("max_steps") or -1
    eff_batch = t["per_device_batch_size"] * t.get("gradient_accumulation_steps", 1)
    total_steps = max_steps if max_steps > 0 else (
        math.ceil(len(train_ds) / eff_batch) * t.get("num_epochs", 1)
    )
    warmup_steps = int(round(t.get("warmup_ratio", 0.0) * total_steps))
    args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        run_name=run_name,
        per_device_train_batch_size=t["per_device_batch_size"],
        per_device_eval_batch_size=t.get("eval_batch_size", t["per_device_batch_size"]),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 1),
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_steps=warmup_steps,
        weight_decay=t.get("weight_decay", 0.0),
        num_train_epochs=t.get("num_epochs", 1),
        max_steps=max_steps,
        logging_steps=t.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=t.get("eval_steps", 200),
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=t.get("save_steps", 500),
        save_total_limit=t.get("save_total_limit", 3),
        bf16=bool(t.get("bf16", True)),
        gradient_checkpointing=bool(t.get("gradient_checkpointing", False)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=t.get("seed", 17),
        report_to=_configure_wandb(t.get("report_to", "none"), run_name),
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=t.get("dataloader_num_workers", 2),
        include_num_input_tokens_seen=True,
    )

    heartbeat = HeartbeatCallback(run_dir)
    n_dec = t.get("decode_eval_examples", 256)
    dec_df = val_df.head(n_dec)
    decode_cb = DecodeEvalCallback(
        tokenizer,
        dec_df[d.get("text_col", "application_text")].tolist(),
        [int(a) for a in dec_df[d.get("label_col", "amount")]],
        batch_size=t.get("decode_eval_batch_size", 32),
        max_new_tokens=t.get("max_new_tokens", 8),
        constrained=bool(t.get("decode_eval_constrained", False)),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=CompletionCollator(pad_token_id=tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=[heartbeat, decode_cb],
    )
    decode_cb.trainer = trainer

    # ---- manifest ----------------------------------------------------------
    import peft as peft_pkg
    import transformers as tf_pkg
    manifest = {
        "run_name": run_name,
        "config_path": str(config_path),
        "resolved_config": cfg,
        "model_repo": model_repo,
        "anchor_token_id": anchor_id,
        "bos_id": tokenizer.bos_token_id,
        "eos_id": tokenizer.eos_token_id,
        "pad_id": tokenizer.pad_token_id,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "effective_batch": t["per_device_batch_size"] * t.get("gradient_accumulation_steps", 1),
        "versions": {"torch": torch.__version__, "transformers": tf_pkg.__version__,
                     "peft": peft_pkg.__version__},
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "git_commit": _git_commit(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    # ---- train -------------------------------------------------------------
    try:
        trainer.train()
    except BaseException:
        heartbeat.mark_failed(traceback.format_exc(limit=20))
        raise

    final_metrics = trainer.evaluate()
    if t.get("save_strategy", "steps") != "no" or cfg.get("minimal_run"):
        adapter_dir = run_dir / "adapter"
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        print(f"[train_subject] adapter saved to {adapter_dir}")
    final = {**{k: _sanitize(v) for k, v in final_metrics.items()},
             "decode": {k: _sanitize(v) for k, v in decode_cb.last_metrics.items()},
             "learning_rate": float(t["learning_rate"]),
             "log_history_tail": trainer.state.log_history[-8:]}
    (run_dir / "final_metrics.json").write_text(json.dumps(final, indent=2, default=str))
    heartbeat._write_status(trainer.state, "completed", extra={"final_eval_loss": _sanitize(final_metrics.get("eval_loss"))})
    print(f"[train_subject] done. final eval: { {k: round(v,4) if isinstance(v,float) else v for k,v in final_metrics.items()} }")
    return final_metrics
