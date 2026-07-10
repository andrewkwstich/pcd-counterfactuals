from __future__ import annotations

import re

import torch
from transformers import LogitsProcessor, LogitsProcessorList

_AMOUNT_RE = re.compile(r"\s*\$?\s*([0-9][0-9,]{0,14})")

_DIGIT_ONLY_RE = re.compile(r"^[0-9]+$")


def parse_amount(text: str) -> int | None:
    m = _AMOUNT_RE.match(text)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return None


def has_trailing_junk(text: str) -> bool:
    m = _AMOUNT_RE.match(text)
    if not m:
        return bool(text.strip())
    return bool(text[m.end():].strip())


_digit_ids_cache: dict[int, list[int]] = {}


def digit_token_ids(tokenizer) -> list[int]:
    key = id(tokenizer)
    if key not in _digit_ids_cache:
        ids = [
            i
            for i in range(len(tokenizer))
            if _DIGIT_ONLY_RE.match(tokenizer.decode([i]))
        ]
        _digit_ids_cache[key] = ids
    return _digit_ids_cache[key]


class DigitsOnlyProcessor(LogitsProcessor):
    def __init__(self, allowed_ids: list[int], eos_token_id: int):
        self.allowed = torch.tensor(sorted(set(allowed_ids) | {eos_token_id}))

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        mask = torch.full_like(scores, float("-inf"))
        allowed = self.allowed.to(scores.device)
        mask[:, allowed] = scores[:, allowed]
        return mask


@torch.no_grad()
def numeric_decode_eval(
    model,
    tokenizer,
    texts: list[str],
    amounts: list[int],
    batch_size: int = 32,
    max_new_tokens: int = 8,
    constrained: bool = False,
) -> dict:
    preds, junk = decode_amounts(
        model, tokenizer, texts,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
        constrained=constrained,
    )
    return score_predictions(preds, amounts, junk)


@torch.no_grad()
def decode_amounts(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 32,
    max_new_tokens: int = 8,
    constrained: bool = False,
) -> tuple[list[int | None], list[bool]]:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos

    gen_cfg = model.generation_config
    gen_cfg.do_sample = False
    gen_cfg.temperature = None
    gen_cfg.top_p = None
    gen_cfg.top_k = None
    gen_cfg.max_length = None

    processors = None
    if constrained:
        processors = LogitsProcessorList(
            [DigitsOnlyProcessor(digit_token_ids(tokenizer), eos)]
        )

    preds: list[int | None] = []
    junk: list[bool] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        encs = tokenizer(chunk, add_special_tokens=False)["input_ids"]
        encs = [[bos] + e for e in encs]
        max_len = max(len(e) for e in encs)
        input_ids = torch.tensor(
            [[pad] * (max_len - len(e)) + e for e in encs], device=device
        )
        attention = torch.tensor(
            [[0] * (max_len - len(e)) + [1] * len(e) for e in encs], device=device
        )
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos,
            pad_token_id=pad,
            use_cache=True,
            logits_processor=processors,
        )
        for row in out[:, max_len:]:
            text = tokenizer.decode(
                row, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            preds.append(parse_amount(text))
            junk.append(has_trailing_junk(text))

    if was_training:
        model.train()
    return preds, junk


def score_predictions(
    preds: list[int | None], amounts: list[int], junk: list[bool] | None = None
) -> dict:
    pairs = [(p, a) for p, a in zip(preds, amounts) if p is not None]
    n, k = len(preds), len(pairs)
    metrics: dict = {"n": n, "parse_rate": (k / n) if n else 0.0}
    if junk is not None:
        metrics["trailing_junk_rate"] = (sum(junk) / n) if n else 0.0
    if k:
        errs = sorted(abs(p - a) for p, a in pairs)
        metrics["mae"] = sum(errs) / k
        metrics["medae"] = float(errs[k // 2])
        mean_a = sum(a for _, a in pairs) / k
        ss_tot = sum((a - mean_a) ** 2 for _, a in pairs)
        ss_res = sum((p - a) ** 2 for p, a in pairs)
        metrics["r2"] = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    else:
        metrics.update(mae=float("nan"), medae=float("nan"), r2=float("nan"))
    return metrics
