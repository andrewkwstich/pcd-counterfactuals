from __future__ import annotations

import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_cfg: dict) -> tuple[Any, Any, str]:
    base = model_cfg["base"]
    mirror = model_cfg.get("tokenizer_mirror") or base
    candidates = [base] if mirror == base else [base, mirror]
    if not os.environ.get("HF_TOKEN") and base.startswith("meta-llama/") and mirror != base:
        print(f"[model_io] HF_TOKEN unset; using ungated mirror {mirror}")
        candidates = [mirror]
    last_err: Exception | None = None
    for repo in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo)
            model = AutoModelForCausalLM.from_pretrained(
                repo, dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            if repo != base:
                print(f"[model_io] WARNING: loaded mirror {repo} instead of {base}")
            return model, tokenizer, repo
        except Exception as e:
            print(f"[model_io] could not load {repo}: {e}")
            last_err = e
    raise RuntimeError(f"failed to load any of {candidates}") from last_err


def load_subject(
    model_cfg: dict, adapter_dir: str | None = None, merge: bool = True
) -> tuple[Any, Any, str]:
    model, tokenizer, repo = load_model_and_tokenizer(model_cfg)
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
        if merge:
            model = model.merge_and_unload()
        print(f"[model_io] adapter loaded from {adapter_dir} (merged={merge})")
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, tokenizer, repo
