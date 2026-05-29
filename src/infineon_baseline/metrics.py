"""Metrics for all three tasks plus a report() helper.

All metric functions are pure: take pandas DataFrames keyed by EXAMPLE_ID,
return a float (or dict for matrix-style outputs).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Block (12 functional categories from generation_rules.md §1)                #
# --------------------------------------------------------------------------- #
# Coarse heuristic mapping. Extend as needed.
_BLOCK_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("LOGISTICS", ("RECEIVE WAFER LOT", "LOT IDENTIFICATION", "LOT RELEASE",
                   "FINAL LOT RELEASE", "SHIP LOT", "PACKAGE PREPARATION")),
    ("INSPECT",   ("INSPECTION", "INSPECT")),
    ("MEASURE",   ("MEASURE",)),
    ("CLEAN",     ("CLEAN", "RCA", "HF DIP", "DRY WAFER", "RINSE")),
    ("DEPOSIT",   ("DEPOSIT", "GROW", "THERMAL OXIDATION", "EPITAXIAL DEPOSITION",
                   "ANNEAL OXIDE", "GATE OXIDE")),
    ("LITHO",     ("SPIN COAT PHOTORESIST", "SOFT BAKE", "ALIGN MASK", "EXPOSE LITHO",
                   "POST EXPOSE BAKE", "DEVELOP PHOTORESIST", "HARD BAKE", "DEVELOP PAD WINDOW",
                   "PAD WINDOW LITHO", "OPEN PAD WINDOW", "OPEN BOND PAD WINDOW")),
    ("ETCH",      ("ETCH", "PASSIVATION ETCH")),
    ("STRIP",     ("STRIP",)),
    ("IMPLANT",   ("IMPLANT", "DRIVE IN DIFFUSION", "RAPID THERMAL ANNEAL", "LIGHT ANNEAL")),
    ("CMP",       ("CMP",)),
    ("VIA_FILL",  ("FILL VIA",)),
    ("TEST",      ("PARAMETRIC TEST", "ELECTRICAL PARAMETRIC TEST", "THRESHOLD VOLTAGE TEST",
                   "BREAKDOWN VOLTAGE TEST", "LEAKAGE TEST", "SWITCHING TEST",
                   "WAFER SORT TEST", "YIELD ANALYSIS", "FINAL ELECTRICAL TEST PREP")),
]


def _step_to_block(step: str) -> str:
    for block, patterns in _BLOCK_PATTERNS:
        for p in patterns:
            if step == p or step.startswith(p) or p in step:
                return block
    return "OTHER"


# Pre-computed lazy dict-style mapping for tests that introspect it.
class _StepToBlock(dict):
    def __missing__(self, key):
        return _step_to_block(key)


STEP_TO_BLOCK = _StepToBlock()


# --------------------------------------------------------------------------- #
# Task 1 metrics                                                              #
# --------------------------------------------------------------------------- #
def top_k_accuracy(preds: pd.DataFrame, gt: pd.DataFrame, k: int) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID")
    hits = 0
    for _, row in merged.iterrows():
        cols = [f"RANK_{i}" for i in range(1, k + 1)]
        if row["NEXT_STEP"] in [row[c] for c in cols]:
            hits += 1
    return hits / len(merged) if len(merged) else 0.0


def mrr(preds: pd.DataFrame, gt: pd.DataFrame) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID")
    total = 0.0
    for _, row in merged.iterrows():
        for rank_i in range(1, 6):
            if row[f"RANK_{rank_i}"] == row["NEXT_STEP"]:
                total += 1.0 / rank_i
                break
    return total / len(merged) if len(merged) else 0.0


# --------------------------------------------------------------------------- #
# Task 2 metrics                                                              #
# --------------------------------------------------------------------------- #
def exact_match_rate(preds: pd.DataFrame, gt: pd.DataFrame) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID")
    matches = (merged["PREDICTED_SEQUENCE"] == merged["GROUND_TRUTH_SEQUENCE"]).sum()
    return float(matches) / len(merged) if len(merged) else 0.0


def _levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, bj in enumerate(b, 1):
            cost = 0 if ai == bj else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def normalized_edit_distance(preds: pd.DataFrame, gt: pd.DataFrame) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID")
    total = 0.0
    for _, row in merged.iterrows():
        p = row["PREDICTED_SEQUENCE"].split("|") if row["PREDICTED_SEQUENCE"] else []
        g = row["GROUND_TRUTH_SEQUENCE"].split("|") if row["GROUND_TRUTH_SEQUENCE"] else []
        if not p and not g:
            continue
        total += _levenshtein(p, g) / max(len(p), len(g))
    return total / len(merged) if len(merged) else 0.0


def token_accuracy(preds: pd.DataFrame, gt: pd.DataFrame) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID")
    correct = total = 0
    for _, row in merged.iterrows():
        p = row["PREDICTED_SEQUENCE"].split("|")
        g = row["GROUND_TRUTH_SEQUENCE"].split("|")
        for a, b in zip(p, g):
            total += 1
            if a == b:
                correct += 1
    return correct / total if total else 0.0


def block_accuracy(preds: pd.DataFrame, gt: pd.DataFrame) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID")
    correct = total = 0
    for _, row in merged.iterrows():
        pb = [STEP_TO_BLOCK[s] for s in row["PREDICTED_SEQUENCE"].split("|")]
        gb = [STEP_TO_BLOCK[s] for s in row["GROUND_TRUTH_SEQUENCE"].split("|")]
        for a, b in zip(pb, gb):
            total += 1
            if a == b:
                correct += 1
    return correct / total if total else 0.0


# --------------------------------------------------------------------------- #
# Task 3 metrics                                                              #
# --------------------------------------------------------------------------- #
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as _sk_confusion,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _binarize_invalid(df: pd.DataFrame) -> pd.Series:
    """Return Series of 1 = invalid (positive class), 0 = valid."""
    return (df["IS_VALID"] == 0).astype(int)


def binary_accuracy(preds: pd.DataFrame, gt: pd.DataFrame) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID", suffixes=("_pred", "_gt"))
    y_pred = (merged["IS_VALID_pred"] == 0).astype(int)
    y_true = (merged["IS_VALID_gt"] == 0).astype(int)
    return float(accuracy_score(y_true, y_pred))


def precision(preds: pd.DataFrame, gt: pd.DataFrame) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID", suffixes=("_pred", "_gt"))
    y_pred = (merged["IS_VALID_pred"] == 0).astype(int)
    y_true = (merged["IS_VALID_gt"] == 0).astype(int)
    return float(precision_score(y_true, y_pred, zero_division=0))


def recall(preds: pd.DataFrame, gt: pd.DataFrame) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID", suffixes=("_pred", "_gt"))
    y_pred = (merged["IS_VALID_pred"] == 0).astype(int)
    y_true = (merged["IS_VALID_gt"] == 0).astype(int)
    return float(recall_score(y_true, y_pred, zero_division=0))


def f1(preds: pd.DataFrame, gt: pd.DataFrame) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID", suffixes=("_pred", "_gt"))
    y_pred = (merged["IS_VALID_pred"] == 0).astype(int)
    y_true = (merged["IS_VALID_gt"] == 0).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))


def confusion_matrix_dict(preds: pd.DataFrame, gt: pd.DataFrame) -> dict[str, int]:
    merged = preds.merge(gt, on="EXAMPLE_ID", suffixes=("_pred", "_gt"))
    y_pred = (merged["IS_VALID_pred"] == 0).astype(int)
    y_true = (merged["IS_VALID_gt"] == 0).astype(int)
    cm = _sk_confusion(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def roc_auc(preds: pd.DataFrame, gt: pd.DataFrame) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID", suffixes=("_pred", "_gt"))
    y_true = (merged["IS_VALID_gt"] == 0).astype(int)
    # Higher SCORE = more confident this row is VALID. To predict the invalid class,
    # use 1 - SCORE as the positive-class likelihood.
    y_score = 1.0 - merged["SCORE"].astype(float)
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def rule_attribution_accuracy(preds: pd.DataFrame, gt: pd.DataFrame) -> float:
    merged = preds.merge(gt, on="EXAMPLE_ID", suffixes=("_pred", "_gt"))
    both_invalid = merged[(merged["IS_VALID_pred"] == 0) & (merged["IS_VALID_gt"] == 0)]
    if len(both_invalid) == 0:
        return 0.0
    matches = (both_invalid["PREDICTED_RULE"] == both_invalid["RULE_VIOLATED"]).sum()
    return float(matches) / len(both_invalid)
