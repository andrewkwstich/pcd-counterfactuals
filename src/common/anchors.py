from __future__ import annotations

import json
from pathlib import Path

ANCHOR_TEXT = "AMOUNT APPROVED: $"

TOKENIZER_REPO = "unsloth/Meta-Llama-3.1-8B-Instruct"

EXPECTED_DOLLAR_TOKEN: int | None = None


def verify_anchor(sample_texts: list[str], amounts: list[int]) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO)

    report: dict = {"tokenizer": TOKENIZER_REPO, "anchor_text": ANCHOR_TEXT, "cases": []}
    dollar_ids = set()
    for text, amount in zip(sample_texts, amounts):
        ids_prompt = tok(text, add_special_tokens=False)["input_ids"]
        ids_full = tok(text + str(amount), add_special_tokens=False)["input_ids"]
        prefix_stable = ids_full[: len(ids_prompt)] == ids_prompt
        last_tok = tok.convert_ids_to_tokens([ids_prompt[-1]])[0]
        dollar_ids.add(ids_prompt[-1])
        report["cases"].append(
            {
                "amount": amount,
                "prefix_stable": prefix_stable,
                "last_prompt_token": last_tok,
                "last_prompt_token_id": ids_prompt[-1],
                "first_answer_tokens": tok.convert_ids_to_tokens(
                    ids_full[len(ids_prompt): len(ids_prompt) + 3]
                ),
            }
        )
    report["anchor_token_id_unique"] = len(dollar_ids) == 1
    report["anchor_token_id"] = dollar_ids.pop() if len(dollar_ids) == 0 else sorted(dollar_ids)[0]
    report["all_prefix_stable"] = all(c["prefix_stable"] for c in report["cases"])
    report["pass"] = report["anchor_token_id_unique"] and report["all_prefix_stable"]
    return report


if __name__ == "__main__":
    import pandas as pd

    apps = pd.read_parquet("data/applications/subject_set.parquet")
    sample = apps.sample(50, random_state=0)
    report = verify_anchor(sample["application_text"].tolist(), sample["amount"].tolist())
    out = Path("artifacts/formula/anchor_check.json")
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
    print("example case:", json.dumps(report["cases"][0], indent=2))
