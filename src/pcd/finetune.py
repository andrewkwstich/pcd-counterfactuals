from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.pcd.data import build_dataloader
from src.pcd.encoder import TopKEncoder
from src.pcd.model import read_middle_activations
from src.pcd.pretrain import Monitor, _git_commit, _sanitize, load_config
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
    lm = cfg.setdefault("lm_data", {})
    lm["num_workers"] = 0
    lm.setdefault("general", {})["source"] = "synthetic"
    cfg.setdefault("readout", {})["max_rows"] = 64
    cfg.setdefault("run", {})
    cfg["run"]["name"] = (cfg["run"].get("name") or "pcd-finetune") + "-minimal"
    cfg["minimal_run"] = True
    return cfg


# --------------------------------------------------------------------------
# QA data
# --------------------------------------------------------------------------

def build_qa_example(
    question: str,
    delta_str: str,
    tokenizer,
    answer_prompt: str = "\nAnswer:",
    eos_id: int | None = None,
) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer(question + answer_prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(" " + delta_str, add_special_tokens=False)["input_ids"]
    if eos_id is not None:
        answer_ids = answer_ids + [eos_id]
    return prompt_ids, answer_ids


def collate_qa(batch: list[dict], pad_id: int) -> dict:
    lens = [len(b["prompt_ids"]) + len(b["answer_ids"]) for b in batch]
    width = max(lens)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        seq = b["prompt_ids"] + b["answer_ids"]
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        labels[i, len(b["prompt_ids"]) : len(seq)] = torch.tensor(
            b["answer_ids"], dtype=torch.long
        )
        attn[i, : len(seq)] = 1
    return {
        "z": torch.stack([b["z"] for b in batch]),
        "input_ids": ids,
        "labels": labels,
        "attention_mask": attn,
    }


class CFQADataset(Dataset):
    def __init__(self, cfg: dict, tokenizer, split: str, limit: int | None = None,
                 seed: int = 0):
        import pandas as pd

        df = pd.read_parquet(
            REPO_ROOT / cfg["path"],
            columns=["question", "delta_str", "split", "stage_c_row_idx", "question_class"],
        )
        df = df[df["split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"no QA rows for split {split!r}")
        if limit is not None and len(df) > limit:
            df = df.sample(n=limit, random_state=seed).reset_index(drop=True)
        self.questions = df["question"].tolist()
        self.deltas = df["delta_str"].tolist()
        self.z_rows = df["stage_c_row_idx"].to_numpy()
        self.z_path = str(REPO_ROOT / cfg["z_path"])
        self._z = None
        self.tokenizer = tokenizer
        self.answer_prompt = cfg.get("answer_prompt", "\nAnswer:")
        self.eos_id = tokenizer.eos_token_id

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, i: int) -> dict:
        if self._z is None:
            self._z = np.load(self.z_path, mmap_mode="r")
        prompt_ids, answer_ids = build_qa_example(
            self.questions[i], self.deltas[i], self.tokenizer,
            self.answer_prompt, self.eos_id,
        )
        z = torch.from_numpy(np.array(self._z[self.z_rows[i]], dtype=np.float32))
        return {"z": z, "prompt_ids": prompt_ids, "answer_ids": answer_ids}


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

def load_finetune_models(model_cfg: dict, subject_adapter: str, decoder_lora_init: str,
                         device) -> tuple:
    from peft import PeftModel

    subject, tokenizer, repo = load_model_and_tokenizer(model_cfg)
    subject = PeftModel.from_pretrained(subject, subject_adapter).merge_and_unload()
    subject.to(device).eval().requires_grad_(False)

    dec_base, _, _ = load_model_and_tokenizer(model_cfg)
    dec_base = PeftModel.from_pretrained(dec_base, subject_adapter).merge_and_unload()
    decoder = PeftModel.from_pretrained(
        dec_base, str(REPO_ROOT / decoder_lora_init), is_trainable=True
    )
    decoder.to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return subject, decoder, tokenizer, repo


def load_frozen_encoder(cfg: dict, device) -> TopKEncoder:
    encoder = TopKEncoder(
        d_model=cfg.get("d_model", 4096),
        n_concepts=cfg.get("n_concepts", 32768),
        k=cfg.get("k", 16),
    )
    state = torch.load(REPO_ROOT / cfg["ckpt"], map_location="cpu", weights_only=True)
    encoder.load_state_dict(state)
    encoder.to(device).eval().requires_grad_(False)
    return encoder


# --------------------------------------------------------------------------
# forward passes
# --------------------------------------------------------------------------

def qa_forward(encoder: TopKEncoder, decoder, embed, batch: dict, device,
               decoder_dtype) -> dict:
    z = batch["z"].to(device)
    ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attn = batch["attention_mask"].to(device)
    b = ids.shape[0]

    with torch.no_grad():
        enc = encoder.encode(z.unsqueeze(1), out_dtype=decoder_dtype)
        embeds = embed(ids)
    inputs_embeds = torch.cat([enc.soft_tokens, embeds], dim=1)
    full_labels = torch.cat([labels.new_full((b, 1), -100), labels], dim=1)
    full_attn = torch.cat([attn.new_ones((b, 1)), attn], dim=1)

    out = decoder(
        inputs_embeds=inputs_embeds,
        attention_mask=full_attn,
        labels=full_labels,
        use_cache=False,
    )
    with torch.no_grad():
        logits = out.logits[:, :-1]
        tgt = full_labels[:, 1:]
        sup = tgt != -100
        tok_correct = ((logits.argmax(-1) == tgt) & sup).sum()
        seq_correct = (((logits.argmax(-1) == tgt) | ~sup).all(dim=-1)).sum()
    return {
        "loss": out.loss,
        "n_answer_tokens": int(sup.sum()),
        "n_tok_correct": int(tok_correct),
        "n_seq": b,
        "n_seq_correct": int(seq_correct),
        "topk_indices": enc.topk_indices,
    }


def lm_forward(subject, encoder: TopKEncoder, decoder, embed, batch: dict,
               read_layer: int, n_middle: int, device, decoder_dtype) -> torch.Tensor:
    suffix_ids = batch["suffix_ids"].to(device)
    suffix_labels = batch["suffix_labels"].to(device)
    b = suffix_ids.shape[0]
    with torch.no_grad():
        a = read_middle_activations(
            subject, batch["subject_input_ids"].to(device), read_layer, n_middle
        )
        enc = encoder.encode(a, out_dtype=decoder_dtype)
        embeds = embed(suffix_ids)
    inputs_embeds = torch.cat([enc.soft_tokens, embeds], dim=1)
    labels = torch.cat([suffix_labels.new_full((b, n_middle), -100), suffix_labels], dim=1)
    attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    out = decoder(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        labels=labels,
        use_cache=False,
    )
    return out.loss


# --------------------------------------------------------------------------
# concept readout
# --------------------------------------------------------------------------

@torch.no_grad()
def write_concept_readout(encoder: TopKEncoder, cfg: dict, out_dir: Path, device) -> dict:
    import pandas as pd

    z = np.load(REPO_ROOT / cfg["z_path"], mmap_mode="r")
    n = min(len(z), int(cfg.get("max_rows") or len(z)))
    labels_path = cfg.get("labels_path")
    labels: dict[str, dict] = {}
    if labels_path and (REPO_ROOT / labels_path).exists():
        labels = json.loads((REPO_ROOT / labels_path).read_text())

    rows_ids, rows_vals = [], []
    bs = 1024
    for lo in range(0, n, bs):
        zb = torch.from_numpy(np.array(z[lo : min(lo + bs, n)], dtype=np.float32)).to(device)
        enc = encoder.encode(zb)
        rows_ids.append(enc.topk_indices.cpu().numpy())
        rows_vals.append(enc.topk_values.float().cpu().numpy())
    ids = np.concatenate(rows_ids)
    vals = np.concatenate(rows_vals)

    def lab(cid: int) -> str | None:
        d = labels.get(str(cid))
        return d.get("description") if d else None

    df = pd.DataFrame({
        "stage_c_row_idx": np.arange(n),
        "concept_ids": [r.tolist() for r in ids],
        "concept_values": [r.tolist() for r in vals],
        "concept_labels": [[lab(c) for c in r] for r in ids],
    })
    manifest_path = cfg.get("manifest_path")
    if manifest_path and (REPO_ROOT / manifest_path).exists():
        m = pd.read_parquet(REPO_ROOT / manifest_path,
                            columns=["row_idx", "app_id", "name_cell"])
        df = df.merge(m, left_on="stage_c_row_idx", right_on="row_idx",
                      how="left").drop(columns=["row_idx"])
    out_path = out_dir / "concept_readout.parquet"
    df.to_parquet(out_path, index=False)

    uniq = len(np.unique(ids))
    labeled = sum(1 for r in ids for c in r if lab(c)) / ids.size if ids.size else 0.0
    return {
        "path": str(out_path.relative_to(REPO_ROOT)),
        "n_rows": int(n),
        "unique_concepts": int(uniq),
        "labeled_slot_frac": round(float(labeled), 4),
    }


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

@torch.no_grad()
def run_validation(encoder, decoder, embed, val_dl, accelerator, decoder_dtype) -> dict:
    decoder.eval()
    tot = torch.zeros(5, device=accelerator.device)
    for batch in val_dl:
        out = qa_forward(encoder, decoder, embed, batch, accelerator.device, decoder_dtype)
        tot += torch.tensor(
            [float(out["loss"]) * out["n_answer_tokens"], out["n_answer_tokens"],
             out["n_tok_correct"], out["n_seq"], out["n_seq_correct"]],
            device=accelerator.device,
        )
    tot = accelerator.reduce(tot, reduction="sum")
    decoder.train()
    loss = float(tot[0] / tot[1]) if tot[1] > 0 else float("nan")
    return {
        "val_qa_loss": round(loss, 4),
        "val_answer_tok_acc": round(float(tot[2] / tot[1]), 4) if tot[1] > 0 else None,
        "val_exact_match": round(float(tot[4] / tot[3]), 4) if tot[3] > 0 else None,
        "val_n": int(tot[3]),
    }


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------

def save_checkpoint(out_dir: Path, tag: str, decoder, step: int, cfg: dict,
                    accelerator) -> None:
    ckpt_dir = out_dir / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    decoder.save_pretrained(ckpt_dir / "decoder_lora")
    (ckpt_dir / "checkpoint_meta.json").write_text(json.dumps({
        "tag": tag, "step": step,
        "encoder_ckpt": cfg["encoder"]["ckpt"],
        "decoder_lora_init": cfg["train"]["decoder_lora_init"],
    }, indent=2))
    accelerator.print(f"[finetune] checkpoint {tag} @ step {step}")


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
    lm_cfg = cfg["lm_data"]
    seq = lm_cfg.get("seq", {"n_prefix": 16, "n_middle": 16, "n_suffix": 16})
    n_middle = seq.get("n_middle", 16)
    read_layer = cfg.get("read_layer", 15)
    seed = t.get("seed", 0)

    ddp = DistributedDataParallelKwargs(broadcast_buffers=False, find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 1),
        kwargs_handlers=[ddp],
    )
    set_seed(seed)
    is_main = accelerator.is_main_process
    rank, world = accelerator.process_index, accelerator.num_processes

    run_name = cfg.get("run", {}).get("name") or "pcd-finetune"
    out_dir = REPO_ROOT / t.get("output_root", "artifacts/pcd") / run_name
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    accelerator.print(f"[finetune] run dir: {out_dir}  |  processes: {world}")

    # ---- models ----
    subject, decoder, tokenizer, repo = load_finetune_models(
        cfg["model"], t["subject_adapter"], t["decoder_lora_init"], accelerator.device
    )
    encoder = load_frozen_encoder(cfg["encoder"], accelerator.device)
    decoder_dtype = next(decoder.parameters()).dtype

    # ---- optimizer ----
    decay, no_decay = [], []
    for _, p in decoder.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": t.get("weight_decay", 0.01)},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=t.get("learning_rate", 1e-4), betas=tuple(t.get("betas", (0.9, 0.999))),
    )

    total_steps = int(t["total_steps"])
    eff_batch = t["per_device_batch_size"] * t.get("gradient_accumulation_steps", 1) * world

    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=0, num_training_steps=total_steps * world
    )

    # ---- data ----
    qa_train = CFQADataset(qa_cfg, tokenizer, split=qa_cfg.get("train_split", "pcd_train"))
    pad_id = tokenizer.pad_token_id
    qa_sampler = DistributedSampler(qa_train, num_replicas=world, rank=rank,
                                    shuffle=True, seed=seed, drop_last=True)
    qa_dl = DataLoader(
        qa_train, batch_size=t["per_device_batch_size"], sampler=qa_sampler,
        collate_fn=lambda b: collate_qa(b, pad_id),
        num_workers=qa_cfg.get("num_workers", 2), pin_memory=True, drop_last=True,
        persistent_workers=qa_cfg.get("num_workers", 2) > 0,
    )
    val_ds = CFQADataset(qa_cfg, tokenizer, split=qa_cfg.get("val_split", "sense1_test"),
                         limit=qa_cfg.get("val_examples", 1024), seed=seed)
    val_shard = torch.utils.data.Subset(val_ds, list(range(rank, len(val_ds), world)))
    val_dl = DataLoader(val_shard, batch_size=t["per_device_batch_size"],
                        collate_fn=lambda b: collate_qa(b, pad_id), num_workers=0)
    lm_dl = build_dataloader(
        lm_cfg, tokenizer, batch_size=t["per_device_batch_size"],
        rank=rank, world_size=world, seed=seed,
    )

    decoder, optim, sched = accelerator.prepare(decoder, optim, sched)
    embed = accelerator.unwrap_model(decoder).get_input_embeddings()

    # ---- W&B ----
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
            "total_steps": total_steps,
            "qa_train_examples": len(qa_train), "val_examples": len(val_ds),
            "config": cfg,
        }, indent=2, default=str))

    milestones = sorted(int(m) for m in t.get("checkpoint_step_milestones", []))
    ms_idx = 0

    # ---- train loop ----
    decoder.train()
    step = 0
    micro = 0
    qa_epoch = 0
    ema: dict[str, float | None] = {"qa": None, "lm": None}
    qa_tok_seen, qa_tok_correct = 0, 0
    qa_iter, lm_iter = iter(qa_dl), iter(lm_dl)
    t_start = time.time()
    accelerator.print(f"[finetune] {total_steps} steps, eff_batch {eff_batch} "
                      f"(50% QA / 50% LM); {len(qa_train):,} QA train examples")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(accelerator.device)

    last_val: dict = {}
    while step < total_steps:
        use_qa = micro % 2 == 0
        if use_qa:
            try:
                batch = next(qa_iter)
            except StopIteration:
                qa_epoch += 1
                qa_sampler.set_epoch(qa_epoch)
                qa_iter = iter(qa_dl)
                batch = next(qa_iter)
        else:
            try:
                batch = next(lm_iter)
            except StopIteration:
                lm_iter = iter(lm_dl)
                batch = next(lm_iter)

        with accelerator.accumulate(decoder):
            if use_qa:
                out = qa_forward(encoder, decoder, embed, batch, accelerator.device,
                                 decoder_dtype)
                loss = out["loss"]
                qa_tok_seen += out["n_answer_tokens"]
                qa_tok_correct += out["n_tok_correct"]
                cur = float(loss.detach().float())
                ema["qa"] = cur if ema["qa"] is None else 0.99 * ema["qa"] + 0.01 * cur
            else:
                loss = lm_forward(subject, encoder, decoder, embed, batch, read_layer,
                                  n_middle, accelerator.device, decoder_dtype)
                cur = float(loss.detach().float())
                ema["lm"] = cur if ema["lm"] is None else 0.99 * ema["lm"] + 0.01 * cur
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                if t.get("max_grad_norm"):
                    accelerator.clip_grad_norm_(decoder.parameters(), t["max_grad_norm"])
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1

                if step % t.get("val_every_steps", 500) == 0 or step == total_steps:
                    last_val = run_validation(encoder, decoder, embed, val_dl,
                                              accelerator, decoder_dtype)
                    accelerator.print(f"[finetune] step {step} val: {last_val}")

                if step % t.get("log_every_steps", 20) == 0 or step == 1:
                    rec = {
                        "qa_loss_ema": round(ema["qa"], 4) if ema["qa"] is not None else None,
                        "lm_loss_ema": round(ema["lm"], 4) if ema["lm"] is not None else None,
                        "train_answer_tok_acc": (round(qa_tok_correct / qa_tok_seen, 4)
                                                 if qa_tok_seen else None),
                        "lr": sched.get_last_lr()[0],
                        "qa_epoch": qa_epoch,
                        "steps_per_sec": round(step / (time.time() - t_start), 3),
                        **last_val,
                    }
                    qa_tok_seen, qa_tok_correct = 0, 0
                    monitor.log(step, total_steps, step * eff_batch,
                                total_steps * eff_batch, rec)
                    if use_wandb:
                        import wandb
                        wandb.log(rec, step=step)
                    accelerator.print(
                        f"[finetune] step {step}/{total_steps} | "
                        f"qa {rec['qa_loss_ema']} lm {rec['lm_loss_ema']} | "
                        f"ans_acc {rec['train_answer_tok_acc']} | lr {rec['lr']:.2e}"
                    )

                while ms_idx < len(milestones) and step >= milestones[ms_idx]:
                    if is_main:
                        save_checkpoint(out_dir, f"ckpt_step{milestones[ms_idx]}",
                                        accelerator.unwrap_model(decoder), step, cfg,
                                        accelerator)
                    ms_idx += 1
        micro += 1

    # ---- final save + concept readout + report ----
    accelerator.wait_for_everyone()
    elapsed = time.time() - t_start
    if is_main:
        save_checkpoint(out_dir, "final", accelerator.unwrap_model(decoder), step, cfg,
                        accelerator)
        readout = write_concept_readout(
            encoder, {**cfg.get("readout", {}), "z_path": qa_cfg["z_path"]},
            out_dir, accelerator.device,
        )
        report = {
            "run_name": run_name, "steps": step,
            "elapsed_seconds": round(elapsed, 1),
            "final_qa_loss_ema": _sanitize(ema["qa"]),
            "final_lm_loss_ema": _sanitize(ema["lm"]),
            **{k: _sanitize(v) for k, v in last_val.items()},
            "qa_epochs": qa_epoch,
            "num_processes": world, "effective_batch": eff_batch,
            "concept_readout": readout,
            "peak_gpu_gib": (round(torch.cuda.max_memory_allocated(accelerator.device) / 2**30, 2)
                             if torch.cuda.is_available() else None),
            "minimal_run": bool(cfg.get("minimal_run")),
        }
        (out_dir / "finetune_report.json").write_text(json.dumps(report, indent=2))
        monitor.status(step, total_steps, step * eff_batch, total_steps * eff_batch,
                       "completed")
        accelerator.print("[finetune] report: " + json.dumps(report, indent=2))
        if use_wandb:
            import wandb
            wandb.finish()
        return report
    return {}
