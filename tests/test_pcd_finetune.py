import torch

from src.pcd.finetune import build_qa_example, collate_qa


class StubTokenizer:
    eos_token_id = 999

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 500 for c in text]}


def test_build_qa_example_prompt_and_answer():
    tok = StubTokenizer()
    prompt, answer = build_qa_example("Q?", "+0", tok, answer_prompt="\nAnswer:",
                                      eos_id=tok.eos_token_id)
    assert prompt == [ord(c) % 500 for c in "Q?\nAnswer:"]
    assert answer == [ord(c) % 500 for c in " +0"] + [999]


def test_build_qa_example_no_eos():
    tok = StubTokenizer()
    _, answer = build_qa_example("Q?", "-5", tok, eos_id=None)
    assert answer == [ord(c) % 500 for c in " -5"]


def _ex(prompt_ids, answer_ids):
    return {"z": torch.zeros(8), "prompt_ids": prompt_ids, "answer_ids": answer_ids}


def test_collate_qa_shapes_and_padding():
    batch = [_ex([1, 2, 3], [10, 11]), _ex([4], [12, 13, 14])]
    out = collate_qa(batch, pad_id=0)
    assert out["input_ids"].shape == (2, 5)
    assert out["z"].shape == (2, 8)
    assert out["input_ids"][0].tolist() == [1, 2, 3, 10, 11]
    assert out["attention_mask"][0].tolist() == [1, 1, 1, 1, 1]
    assert out["input_ids"][1].tolist() == [4, 12, 13, 14, 0]
    assert out["attention_mask"][1].tolist() == [1, 1, 1, 1, 0]


def test_collate_qa_labels_only_on_answer():
    batch = [_ex([1, 2, 3], [10, 11]), _ex([4], [12, 13, 14])]
    out = collate_qa(batch, pad_id=0)
    assert out["labels"][0].tolist() == [-100, -100, -100, 10, 11]
    assert out["labels"][1].tolist() == [-100, 12, 13, 14, -100]


def test_collate_qa_single_element():
    out = collate_qa([_ex([7, 8], [9])], pad_id=0)
    assert out["input_ids"].tolist() == [[7, 8, 9]]
    assert out["labels"].tolist() == [[-100, -100, 9]]
    assert out["attention_mask"].tolist() == [[1, 1, 1]]
