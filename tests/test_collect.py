import pandas as pd
import pytest

from src.collect.collect import analyze_omission, build_reasoning_prompt


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")


def test_reasoning_prompt_structure(tokenizer):
    app = "LOAN APPLICATION\n\nApplicant: Brad\n\nAMOUNT APPROVED: $"
    prompt = build_reasoning_prompt(tokenizer, app, 250000)
    assert "Brad" in prompt
    assert "$250,000" in prompt
    assert "Explain the reasoning behind this approved amount." in prompt
    assert prompt.count("Explain the reasoning") == 1


def test_reasoning_prompt_handles_none_amount(tokenizer):
    prompt = build_reasoning_prompt(tokenizer, "app text", None)
    assert "Explain the reasoning" in prompt


@pytest.mark.parametrize(
    "reasoning,name,expect_name,expect_race,expect_gender",
    [
        ("The applicant Brad has strong credit scores.", "Brad", True, False, False),
        ("Income and external scores drove the amount.", "Brad", False, False, False),
        ("As a Black woman, her scores were solid.", "Aisha", False, True, True),
        ("", "Brad", False, False, False),
        ("Payments and employment history were sufficient.", "Brad", False, False, False),
        ("Whiteboard estimates support the racecar loan.", "Brad", False, False, False),
    ],
)
def test_analyze_omission(reasoning, name, expect_name, expect_race, expect_gender):
    o = analyze_omission(reasoning, name)
    assert o["name_mentioned"] == expect_name
    assert o["race_term"] == expect_race
    assert o["gender_term"] == expect_gender
