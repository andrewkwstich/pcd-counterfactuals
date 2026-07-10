import random

from src.pcd.data import (
    PretrainCollator,
    chunk_window,
    decision_line_positions,
    is_finance,
    split_segments,
)


def test_chunk_window_too_short_returns_none():
    assert chunk_window(list(range(10)), 16, 16, 16, random.Random(0)) is None


def test_chunk_window_head_starts_at_zero():
    ids = list(range(60))
    start, win = chunk_window(ids, 16, 16, 16, random.Random(0), window="head")
    assert start == 0 and win == list(range(48))


def test_chunk_window_random_in_range():
    ids = list(range(100))
    for seed in range(20):
        start, win = chunk_window(ids, 16, 16, 16, random.Random(seed))
        assert 0 <= start <= 100 - 48
        assert win == ids[start : start + 48]


def test_split_segments_no_mask():
    win = list(range(48))
    seg = split_segments(win, 16, 16, 16, "general")
    assert seg.prefix == list(range(16))
    assert seg.middle == list(range(16, 32))
    assert seg.suffix == list(range(32, 48))
    assert seg.suffix_mask == [1] * 16
    assert seg.source == "general"


def test_split_segments_masks_overlapping_positions():
    win = list(range(48))
    mask_positions = set(range(40, 60))
    seg = split_segments(win, 16, 16, 16, "application",
                         mask_positions=mask_positions, window_start=0)
    assert seg.suffix_mask == [1] * 8 + [0] * 8


def test_split_segments_respects_window_start():
    win = list(range(100, 148))
    mask_positions = set(range(144, 200))
    seg = split_segments(win, 16, 16, 16, "application",
                         mask_positions=mask_positions, window_start=100)
    assert seg.suffix_mask == [1] * 12 + [0] * 4


def test_decision_line_positions():
    app_ids = [1, 2, 3, 99, 400, 5, 6]
    pos = decision_line_positions(app_ids, anchor_token_id=400, anchor_len=2)
    assert pos == {3, 4, 5, 6}


def test_decision_line_positions_no_anchor():
    assert decision_line_positions([1, 2, 3], anchor_token_id=400, anchor_len=2) == set()


def test_is_finance():
    assert is_finance("The lender reviewed the loan and credit history of the borrower.")
    assert is_finance("Compare mortgage rates: a 30-year mortgage or refinancing your home loan.")
    assert not is_finance("The cat sat quietly on the warm windowsill all afternoon.")
    assert not is_finance("A single loan is not enough by itself here.")
    assert not is_finance("The bank of the river drew interest from tourists seeking a good income.")


def test_collator_builds_subject_and_masked_labels():
    collate = PretrainCollator(instruct_prefix_ids=[100, 101], n_middle=2)
    batch = [
        {"prefix": [1, 2], "middle": [3, 4], "suffix": [5, 6], "suffix_mask": [1, 0], "source": 0},
        {"prefix": [7, 8], "middle": [9, 10], "suffix": [11, 12], "suffix_mask": [1, 1], "source": 2},
    ]
    out = collate(batch)
    assert out["subject_input_ids"].tolist() == [[100, 101, 1, 2, 3, 4], [100, 101, 7, 8, 9, 10]]
    assert out["suffix_ids"].tolist() == [[5, 6], [11, 12]]
    assert out["suffix_labels"].tolist() == [[5, -100], [11, 12]]
    assert out["source"].tolist() == [0, 2]
