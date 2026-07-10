from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from src.pcd.data import build_dataloader
from src.pcd.encoder import TopKEncoder
from src.pcd.model import PCDModel, load_pcd_models, read_middle_activations

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_minimal_run(cfg: dict) -> dict:
    t = cfg.setdefault("train", {})
    t["max_tokens"] = min(t.get("max_tokens") or 20000, 20000)
    t["per_device_batch_size"] = min(t.get("per_device_batch_size") or 4, 4)
    t["gradient_accumulation_steps"] = 1
    t["log_every_steps"] = 1
    t["checkpoint_token_milestones"] = []
    t["report_to"] = "none"
    d = cfg.setdefault("data", {})
    d["num_workers"] = 0
    d.setdefault("general", {})["source"] = "synthetic"
    d.setdefault("mixture", {"general": 0.7, "domain": 0.15, "application": 0.15})
    cfg.setdefault("run", {})
    cfg["run"]["name"] = (cfg["run"].get("name") or "pcd-pretrain") + "-minimal"
    cfg["minimal_run"] = True
    return cfg


# --------------------------------------------------------------------------
# monitoring
# --------------------------------------------------------------------------

def _sanitize(v: Any) -> Any:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


class Monitor:
    def __init__(self, run_dir: Path, enabled: bool) -> None:
        self.enabled = enabled
        self.metrics_path = run_dir / "metrics.jsonl"
        self.status_path = run_dir / "status.json"
        self.t0 = time.time()
        self.last: dict[str, Any] = {}

    def log(self, step: int, total_steps: int, tokens: int, max_tokens: int, rec: dict) -> None:
        if not self.enabled:
            return
        rec = {k: _sanitize(v) for k, v in rec.items()}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                "step": step, "tokens": tokens, **rec}) + "\n")
        self.last.update(rec)
        self.status(step, total_steps, tokens, max_tokens, "running")

    def status(self, step: int, total_steps: int, tokens: int, max_tokens: int, state: str) -> None:
        if not self.enabled:
            return
        elapsed = time.time() - self.t0
        rate = tokens / elapsed if elapsed > 0 else 0.0
        eta = (max_tokens - tokens) / rate if rate > 0 and state == "running" else None
        payload = {
            "state": state,
            "step": step,
            "total_steps": total_steps,
            "tokens": tokens,
            "max_tokens": max_tokens,
            "pct_complete": round(100.0 * tokens / max_tokens, 2) if max_tokens else None,
            "tokens_per_sec": round(rate, 1),
            "eta_seconds": round(eta) if eta is not None else None,
            "elapsed_seconds": round(elapsed),
            "pid": os.getpid(),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **self.last,
        }
        self.status_path.write_text(json.dumps(payload, indent=2))


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------

def save_checkpoint(out_dir: Path, tag: str, encoder: TopKEncoder, decoder, tokens: int,
                    step: int, cfg: dict, accelerator) -> None:
    ckpt_dir = out_dir / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), ckpt_dir / "encoder.pt")
    decoder.save_pretrained(ckpt_dir / "decoder_lora")
    (ckpt_dir / "checkpoint_meta.json").write_text(json.dumps({
        "tag": tag, "tokens": tokens, "step": step,
        "n_concepts": encoder.n_concepts, "k": encoder.k,
        **encoder.activity_stats(),
    }, indent=2))
    accelerator.print(f"[pretrain] checkpoint {tag} @ {tokens/1e6:.2f}M tokens")


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
    seq = cfg["data"].get("seq", {"n_prefix": 16, "n_middle": 16, "n_suffix": 16})
    n_middle, n_suffix = seq.get("n_middle", 16), seq.get("n_suffix", 16)
    read_layer = cfg.get("read_layer", 15)
    seed = t.get("seed", 0)

    ddp = DistributedDataParallelKwargs(broadcast_buffers=False, find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 1),
        kwargs_handlers=[ddp],
    )
    set_seed(seed)
    is_main = accelerator.is_main_process

    run_name = cfg.get("run", {}).get("name") or "pcd-pretrain"
    out_dir = REPO_ROOT / t.get("output_root", "artifacts/pcd") / run_name
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    accelerator.print(f"[pretrain] run dir: {out_dir}  |  processes: {accelerator.num_processes}")

    # ---- models ----
    subject, decoder, tokenizer, repo = load_pcd_models(
        cfg["model"], t["subject_adapter"], t["lora"], device=accelerator.device
    )
    encoder = TopKEncoder(
        d_model=subject.config.hidden_size,
        n_concepts=cfg["encoder"].get("n_concepts", 32768),
        k=cfg["encoder"].get("k", 16),
        aux_k=cfg["encoder"].get("aux_k", 500),
        aux_coef=cfg["encoder"].get("aux_coef", 1e-4),
        dead_window_tokens=cfg["encoder"].get("dead_window_tokens", 1_000_000),
    ).to(accelerator.device)
    pcd = PCDModel(encoder, decoder, n_middle=n_middle, n_suffix=n_suffix)

    # ---- optimizer ----
    decay, no_decay = [], []
    for name, p in pcd.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": t.get("weight_decay", 0.01)},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=t.get("learning_rate", 1e-4), betas=tuple(t.get("betas", (0.9, 0.999))),
    )

    # ---- token budget -> optimizer steps ----
    eff_batch = (t["per_device_batch_size"] * t.get("gradient_accumulation_steps", 1)
                 * accelerator.num_processes)
    accounting = t.get("token_accounting", "encoder")
    tokens_per_example = {"encoder": n_middle, "suffix": n_suffix,
                          "total": seq.get("n_prefix", 16) + n_middle + n_suffix}[accounting]
    tokens_per_opt_step = eff_batch * tokens_per_example
    max_tokens = int(t["max_tokens"])
    total_steps = max(1, math.ceil(max_tokens / tokens_per_opt_step))

    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=0,
        num_training_steps=total_steps * accelerator.num_processes,
    )

    dl = build_dataloader(
        cfg["data"], tokenizer, batch_size=t["per_device_batch_size"],
        rank=accelerator.process_index, world_size=accelerator.num_processes, seed=seed,
    )
    pcd, optim, sched = accelerator.prepare(pcd, optim, sched)

    # ---- W&B (main only) ----
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
            "num_processes": accelerator.num_processes, "effective_batch": eff_batch,
            "tokens_per_opt_step": tokens_per_opt_step, "total_steps": total_steps,
            "max_tokens": max_tokens, "token_accounting": accounting,
            "config": cfg,
        }, indent=2, default=str))

    milestones = sorted(int(m) for m in t.get("checkpoint_token_milestones", []))
    ms_idx = 0

    # ---- train loop ----
    pcd.train()
    tokens = 0
    step = 0
    ema_lm: float | None = None
    t_start = time.time()
    win_active = torch.zeros(encoder.n_concepts, dtype=torch.bool, device=accelerator.device)
    data_iter = iter(dl)
    accelerator.print(f"[pretrain] target {max_tokens/1e6:.1f}M tokens "
                      f"({accounting}) -> {total_steps} steps; eff_batch {eff_batch}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(accelerator.device)
    done = False
    while not done:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        with accelerator.accumulate(pcd):
            with torch.no_grad():
                a = read_middle_activations(
                    subject, batch["subject_input_ids"].to(accelerator.device),
                    read_layer, n_middle,
                )
            out = pcd(a, batch["suffix_ids"].to(accelerator.device),
                      batch["suffix_labels"].to(accelerator.device))
            accelerator.backward(out["loss"])
            win_active |= encoder.batch_active_mask(out["topk_indices"])
            if accelerator.sync_gradients:
                if t.get("max_grad_norm"):
                    accelerator.clip_grad_norm_(pcd.parameters(), t["max_grad_norm"])
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)

                global_active = accelerator.reduce(win_active.int(), reduction="max").bool()
                encoder.update_activity(global_active, n_tokens=tokens_per_opt_step)
                win_active.zero_()

                step += 1
                tokens += tokens_per_opt_step

                lm_cur = float(out["lm_loss"].float())
                ema_lm = lm_cur if ema_lm is None else 0.99 * ema_lm + 0.01 * lm_cur

                if step % t.get("log_every_steps", 20) == 0 or step == 1:
                    stats = encoder.activity_stats()
                    rec = {
                        "loss": float(out["loss"].detach().float()),
                        "lm_loss": lm_cur,
                        "lm_loss_ema": round(ema_lm, 4),
                        "aux_loss": float(out["aux_loss"].float()),
                        "lr": sched.get_last_lr()[0],
                        "frac_alive": stats["frac_alive"],
                        "n_dead": stats["n_dead"],
                        "tokens_per_sec": round(tokens / (time.time() - t_start), 1),
                    }
                    monitor.log(step, total_steps, tokens, max_tokens, rec)
                    if use_wandb:
                        import wandb
                        wandb.log({**rec, "tokens": tokens}, step=step)
                    accelerator.print(
                        f"[pretrain] step {step}/{total_steps} | {tokens/1e6:.2f}M tok | "
                        f"loss {rec['loss']:.4f} lm {rec['lm_loss']:.4f} aux {rec['aux_loss']:.2e} | "
                        f"alive {stats['frac_alive']:.3f} | {rec['tokens_per_sec']:.0f} tok/s"
                    )

                while ms_idx < len(milestones) and tokens >= milestones[ms_idx]:
                    if is_main:
                        save_checkpoint(out_dir, f"ckpt_{milestones[ms_idx]//1_000_000}M",
                                        encoder, accelerator.unwrap_model(decoder),
                                        tokens, step, cfg, accelerator)
                    ms_idx += 1

                if tokens >= max_tokens:
                    done = True

    # ---- final save + report ----
    accelerator.wait_for_everyone()
    elapsed = time.time() - t_start
    if is_main:
        save_checkpoint(out_dir, "final", encoder, accelerator.unwrap_model(decoder),
                        tokens, step, cfg, accelerator)
        report = {
            "run_name": run_name, "tokens": tokens, "steps": step,
            "elapsed_seconds": round(elapsed, 1),
            "tokens_per_sec": round(tokens / elapsed, 1) if elapsed else None,
            "final_loss": _sanitize(float(out["loss"].detach().float())),
            "final_lm_loss": _sanitize(float(out["lm_loss"].float())),
            "num_processes": accelerator.num_processes,
            "effective_batch": eff_batch,
            "token_accounting": accounting,
            "peak_gpu_gib": (round(torch.cuda.max_memory_allocated(accelerator.device) / 2**30, 2)
                             if torch.cuda.is_available() else None),
            **encoder.activity_stats(),
            "minimal_run": bool(cfg.get("minimal_run")),
        }
        (out_dir / "pretrain_report.json").write_text(json.dumps(report, indent=2))
        monitor.status(step, total_steps, tokens, max_tokens, "completed")
        accelerator.print("[pretrain] report: " + json.dumps(report, indent=2))
        if use_wandb:
            import wandb
            wandb.finish()
        return report
    return {}
