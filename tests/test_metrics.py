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


from infineon_baseline.metrics import (
    binary_accuracy, precision, recall, f1, confusion_matrix_dict,
    roc_auc, rule_attribution_accuracy,
)


def test_binary_accuracy_simple():
    preds = pd.DataFrame([
        {"EXAMPLE_ID": "a", "IS_VALID": 1, "SCORE": 0.9, "PREDICTED_RULE": ""},
        {"EXAMPLE_ID": "b", "IS_VALID": 0, "SCORE": 0.1, "PREDICTED_RULE": "RULE_X"},
    ])
    gt = pd.DataFrame([
        {"EXAMPLE_ID": "a", "IS_VALID": 1, "RULE_VIOLATED": ""},
        {"EXAMPLE_ID": "b", "IS_VALID": 1, "RULE_VIOLATED": ""},  # mispredicted
    ])
    assert binary_accuracy(preds, gt) == 0.5


def test_precision_recall_f1_consistency():
    # 1 TP, 1 FP, 1 FN  → P=0.5, R=0.5, F1=0.5
    preds = pd.DataFrame([
        {"EXAMPLE_ID": "a", "IS_VALID": 0, "SCORE": 0.1, "PREDICTED_RULE": "X"},  # TP (correctly invalid)
        {"EXAMPLE_ID": "b", "IS_VALID": 0, "SCORE": 0.2, "PREDICTED_RULE": "X"},  # FP
        {"EXAMPLE_ID": "c", "IS_VALID": 1, "SCORE": 0.9, "PREDICTED_RULE": ""},   # FN (truly invalid)
    ])
    gt = pd.DataFrame([
        {"EXAMPLE_ID": "a", "IS_VALID": 0, "RULE_VIOLATED": "X"},
        {"EXAMPLE_ID": "b", "IS_VALID": 1, "RULE_VIOLATED": ""},
        {"EXAMPLE_ID": "c", "IS_VALID": 0, "RULE_VIOLATED": "Y"},
    ])
    # Positive class = invalid (IS_VALID==0).
    assert precision(preds, gt) == 0.5
    assert recall(preds, gt) == 0.5
    assert abs(f1(preds, gt) - 0.5) < 1e-9


def test_roc_auc_uses_score_column():
    preds = pd.DataFrame([
        {"EXAMPLE_ID": "a", "IS_VALID": 1, "SCORE": 0.9, "PREDICTED_RULE": ""},
        {"EXAMPLE_ID": "b", "IS_VALID": 0, "SCORE": 0.1, "PREDICTED_RULE": "X"},
        {"EXAMPLE_ID": "c", "IS_VALID": 1, "SCORE": 0.8, "PREDICTED_RULE": ""},
        {"EXAMPLE_ID": "d", "IS_VALID": 0, "SCORE": 0.2, "PREDICTED_RULE": "X"},
    ])
    gt = pd.DataFrame([
        {"EXAMPLE_ID": "a", "IS_VALID": 1, "RULE_VIOLATED": ""},
        {"EXAMPLE_ID": "b", "IS_VALID": 0, "RULE_VIOLATED": "X"},
        {"EXAMPLE_ID": "c", "IS_VALID": 1, "RULE_VIOLATED": ""},
        {"EXAMPLE_ID": "d", "IS_VALID": 0, "RULE_VIOLATED": "X"},
    ])
    # Perfect separation by SCORE → AUC = 1.0
    assert roc_auc(preds, gt) == 1.0


def test_rule_attribution_accuracy():
    preds = pd.DataFrame([
        {"EXAMPLE_ID": "a", "IS_VALID": 0, "SCORE": 0.1, "PREDICTED_RULE": "RULE_X"},
        {"EXAMPLE_ID": "b", "IS_VALID": 0, "SCORE": 0.1, "PREDICTED_RULE": "RULE_X"},
    ])
    gt = pd.DataFrame([
        {"EXAMPLE_ID": "a", "IS_VALID": 0, "RULE_VIOLATED": "RULE_X"},  # correct
        {"EXAMPLE_ID": "b", "IS_VALID": 0, "RULE_VIOLATED": "RULE_Y"},  # incorrect
    ])
    assert rule_attribution_accuracy(preds, gt) == 0.5


def test_confusion_matrix_dict_has_tp_fp_fn_tn():
    preds = pd.DataFrame([{"EXAMPLE_ID": "a", "IS_VALID": 1, "SCORE": 0.9, "PREDICTED_RULE": ""}])
    gt = pd.DataFrame([{"EXAMPLE_ID": "a", "IS_VALID": 1, "RULE_VIOLATED": ""}])
    cm = confusion_matrix_dict(preds, gt)
    assert set(cm.keys()) == {"tp", "fp", "fn", "tn"}
