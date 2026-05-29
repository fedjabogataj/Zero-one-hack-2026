import pandas as pd

from infineon_baseline.metrics import (
    top_k_accuracy, mrr, exact_match_rate, normalized_edit_distance,
    token_accuracy, block_accuracy, STEP_TO_BLOCK,
)


def test_top_k_accuracy_known_case():
    preds = pd.DataFrame([
        {"EXAMPLE_ID": "a", "RANK_1": "X", "RANK_2": "Y", "RANK_3": "Z", "RANK_4": "Q", "RANK_5": "R"},
    ])
    gt = pd.DataFrame([{"EXAMPLE_ID": "a", "NEXT_STEP": "Y"}])
    assert top_k_accuracy(preds, gt, k=2) == 1.0
    assert top_k_accuracy(preds, gt, k=1) == 0.0


def test_mrr_uses_inverse_rank_of_first_hit():
    preds = pd.DataFrame([
        {"EXAMPLE_ID": "a", "RANK_1": "X", "RANK_2": "Y", "RANK_3": "Z", "RANK_4": "Q", "RANK_5": "R"},
    ])
    gt = pd.DataFrame([{"EXAMPLE_ID": "a", "NEXT_STEP": "Y"}])
    assert mrr(preds, gt) == 0.5


def test_exact_match_rate():
    preds = pd.DataFrame([
        {"EXAMPLE_ID": "a", "PREDICTED_SEQUENCE": "A|B|C"},
        {"EXAMPLE_ID": "b", "PREDICTED_SEQUENCE": "X|Y"},
    ])
    gt = pd.DataFrame([
        {"EXAMPLE_ID": "a", "GROUND_TRUTH_SEQUENCE": "A|B|C"},
        {"EXAMPLE_ID": "b", "GROUND_TRUTH_SEQUENCE": "X|Z"},
    ])
    assert exact_match_rate(preds, gt) == 0.5


def test_normalized_edit_distance():
    # Two identical: 0.0 ; one off-by-one of length 3: 1/3.
    preds = pd.DataFrame([
        {"EXAMPLE_ID": "a", "PREDICTED_SEQUENCE": "A|B|C"},
        {"EXAMPLE_ID": "b", "PREDICTED_SEQUENCE": "A|X|C"},
    ])
    gt = pd.DataFrame([
        {"EXAMPLE_ID": "a", "GROUND_TRUTH_SEQUENCE": "A|B|C"},
        {"EXAMPLE_ID": "b", "GROUND_TRUTH_SEQUENCE": "A|B|C"},
    ])
    val = normalized_edit_distance(preds, gt)
    assert 0.15 < val < 0.2  # (0 + 1/3) / 2 ≈ 0.1666


def test_token_accuracy_overlap_region():
    preds = pd.DataFrame([
        {"EXAMPLE_ID": "a", "PREDICTED_SEQUENCE": "A|B|C|D"},
    ])
    gt = pd.DataFrame([
        {"EXAMPLE_ID": "a", "GROUND_TRUTH_SEQUENCE": "A|B|X|D"},  # 3/4 match
    ])
    assert token_accuracy(preds, gt) == 0.75


def test_block_accuracy_uses_step_to_block_mapping():
    preds = pd.DataFrame([
        {"EXAMPLE_ID": "a", "PREDICTED_SEQUENCE": "PRE CLEAN WAFER|OXIDE ETCH"},
    ])
    gt = pd.DataFrame([
        # Different step strings but same blocks (Cleaning, Etch)
        {"EXAMPLE_ID": "a", "GROUND_TRUTH_SEQUENCE": "RCA CLEAN 1|OXIDE ETCH DRY"},
    ])
    # block_accuracy compares the sequence of block names → both [CLEAN, ETCH]
    assert block_accuracy(preds, gt) == 1.0


def test_step_to_block_covers_basic_categories():
    # Smoke check that the mapping covers a handful of representative steps.
    assert STEP_TO_BLOCK["RECEIVE WAFER LOT"] == "LOGISTICS"
    assert STEP_TO_BLOCK["PRE CLEAN WAFER"] == "CLEAN"
    assert STEP_TO_BLOCK["OXIDE ETCH"] == "ETCH"
    assert STEP_TO_BLOCK["DEPOSIT POLYSILICON"] == "DEPOSIT"
    assert STEP_TO_BLOCK["IMPLANT WELL"] == "IMPLANT"
    assert STEP_TO_BLOCK["WAFER SORT TEST"] == "TEST"
    assert STEP_TO_BLOCK["SHIP LOT"] == "LOGISTICS"
