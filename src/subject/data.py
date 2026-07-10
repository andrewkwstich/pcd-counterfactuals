from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

ANCHOR_TOKEN_ID = 400


class SubjectDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        text_col: str = "application_text",
        label_col: str = "amount",
        anchor_token_id: int = ANCHOR_TOKEN_ID,
    ) -> None:
        self.examples: list[dict[str, list[int]]] = []
        texts = df[text_col].tolist()
        amounts = df[label_col].tolist()
        enc_prompts = tokenizer(texts, add_special_tokens=False)["input_ids"]
        enc_answers = tokenizer(
            [str(int(a)) for a in amounts], add_special_tokens=False
        )["input_ids"]
        bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
        for prompt_ids, answer_ids, amount in zip(enc_prompts, enc_answers, amounts):
            if prompt_ids[-1] != anchor_token_id:
                raise ValueError(
                    f"prompt does not end with anchor token {anchor_token_id}; "
                    f"got {prompt_ids[-6:]}"
                )
            input_ids = [bos] + prompt_ids + answer_ids + [eos]
            labels = [-100] * (1 + len(prompt_ids)) + answer_ids + [eos]
            self.examples.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "amount": int(amount),
                    "prompt_len": 1 + len(prompt_ids),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


@dataclass
class CompletionCollator:
    pad_token_id: int

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids, labels, attention = [], [], []
        for ex in batch:
            ids, labs = ex["input_ids"], ex["labels"]
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            labels.append(labs + [-100] * pad)
            attention.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


def load_split(
    path: str,
    split_tag: str,
    max_examples: int | None = None,
    split_col: str = "split_tag",
    seed: int = 0,
) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df[split_col] == split_tag]
    if len(df) == 0:
        raise ValueError(f"no rows with {split_col}=={split_tag!r} in {path}")
    if max_examples is not None and len(df) > max_examples:
        df = df.sample(max_examples, random_state=seed)
    return df.reset_index(drop=True)
