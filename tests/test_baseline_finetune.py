import torch

from src.baseline.finetune import build_baseline_example, collate_baseline


class StubTokenizer:
    eos_token_id = 999

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 500 for c in text]}


def test_build_example_no_truncation():
    tok = StubTokenizer()
    ctx = [1, 2, 3, 4, 5]
    prompt, answer = build_baseline_example(ctx, "Q?", "+0", tok,
                                            answer_prompt="\nA:", max_len=64,
                                            eos_id=tok.eos_token_id)
    assert prompt == ctx + [ord(c) % 500 for c in "Q?\nA:"]
    assert answer == [ord(c) % 500 for c in " +0"] + [999]


def test_build_example_truncates_context_keeps_qa_and_answer():
    tok = StubTokenizer()
    ctx = list(range(1, 101))
    q = "Q?"; ap = "\nA:"
    qa_len = len(q + ap)
    ans = [ord(c) % 500 for c in " +5"] + [999]
    max_len = 20
    prompt, answer = build_baseline_example(ctx, q, "+5", tok, answer_prompt=ap,
                                            max_len=max_len, eos_id=tok.eos_token_id)
    budget = max_len - qa_len - len(ans)
    assert len(prompt) + len(answer) <= max_len
    assert prompt[:budget] == ctx[:budget]
    assert prompt[budget:] == [ord(c) % 500 for c in q + ap]
    assert answer == ans


def test_build_example_answer_never_truncated():
    tok = StubTokenizer()
    ctx = list(range(1, 51))
    _, answer = build_baseline_example(ctx, "Question here", "-123400", tok,
                                       max_len=8, eos_id=tok.eos_token_id)
    assert answer == [ord(c) % 500 for c in " -123400"] + [999]


def _ex(p, a):
    return {"prompt_ids": p, "answer_ids": a}


def test_collate_shapes_padding_and_labels():
    batch = [_ex([1, 2, 3], [10, 11]), _ex([4], [12, 13, 14])]
    out = collate_baseline(batch, pad_id=0)
    assert out["input_ids"].shape == (2, 5)
    assert out["input_ids"][0].tolist() == [1, 2, 3, 10, 11]
    assert out["input_ids"][1].tolist() == [4, 12, 13, 14, 0]
    assert out["attention_mask"][1].tolist() == [1, 1, 1, 1, 0]
    assert out["labels"][0].tolist() == [-100, -100, -100, 10, 11]
    assert out["labels"][1].tolist() == [-100, 12, 13, 14, -100]


def test_collate_single():
    out = collate_baseline([_ex([7, 8], [9])], pad_id=0)
    assert out["input_ids"].tolist() == [[7, 8, 9]]
    assert out["labels"].tolist() == [[-100, -100, 9]]
