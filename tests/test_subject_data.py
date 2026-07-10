import pandas as pd
import pytest

from src.subject.data import ANCHOR_TOKEN_ID, CompletionCollator, SubjectDataset
from src.subject.decode_eval import parse_amount

TEXT = (
    "LOAN APPLICATION\n\nApplicant: Testname\nAnnual income: $50,000\n\n"
    "AMOUNT APPROVED: $"
)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")


@pytest.fixture(scope="module")
def dataset(tokenizer):
    df = pd.DataFrame(
        {"application_text": [TEXT, TEXT], "amount": [734600, 7400]}
    )
    return SubjectDataset(df, tokenizer)


def test_masking_supervises_exactly_answer_plus_eos(tokenizer, dataset):
    for ex, amount in zip(dataset.examples, [734600, 7400]):
        supervised = [t for t in ex["labels"] if t != -100]
        assert supervised[-1] == tokenizer.eos_token_id
        assert tokenizer.decode(supervised[:-1]) == str(amount)
        assert ex["labels"][: ex["prompt_len"]] == [-100] * ex["prompt_len"]
        assert ex["labels"][ex["prompt_len"]] != -100
        assert ex["input_ids"][ex["prompt_len"] - 1] == ANCHOR_TOKEN_ID
        assert ex["input_ids"][0] == tokenizer.bos_token_id
        assert ex["input_ids"][-1] == tokenizer.eos_token_id
        assert len(ex["labels"]) == len(ex["input_ids"])


def test_missing_anchor_raises(tokenizer):
    df = pd.DataFrame({"application_text": ["AMOUNT APPROVED: "], "amount": [1000]})
    with pytest.raises(ValueError, match="anchor"):
        SubjectDataset(df, tokenizer)


def test_collator_pads_right(tokenizer, dataset):
    collator = CompletionCollator(pad_token_id=tokenizer.pad_token_id)
    batch = collator(list(dataset.examples))
    n0, n1 = (len(ex["input_ids"]) for ex in dataset.examples)
    width = batch["input_ids"].shape[1]
    assert width == max(n0, n1)
    for row, n in zip(range(2), (n0, n1)):
        assert batch["attention_mask"][row, :n].all()
        assert not batch["attention_mask"][row, n:].any()
        assert (batch["labels"][row, n:] == -100).all()
        assert (batch["input_ids"][row, n:] == tokenizer.pad_token_id).all()


@pytest.mark.parametrize(
    "text,expected",
    [("734600", 734600), (" $1,250,000 blah", 1250000), ("", None), ("N/A", None)],
)
def test_parse_amount(text, expected):
    assert parse_amount(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [("734600", False), ("734600 approved", True), ("  123\n", False),
     ("ok", True), ("", False)],
)
def test_has_trailing_junk(text, expected):
    from src.subject.decode_eval import has_trailing_junk

    assert has_trailing_junk(text) == expected


def test_digit_token_ids(tokenizer):
    from src.subject.decode_eval import digit_token_ids

    ids = digit_token_ids(tokenizer)
    assert ids, "no digit tokens found"
    for i in ids[:50] + ids[-50:]:
        assert tokenizer.decode([i]).isdigit()
    for s in ("734", "600", "0", "999"):
        enc = tokenizer(s, add_special_tokens=False)["input_ids"]
        assert all(t in set(ids) for t in enc)
