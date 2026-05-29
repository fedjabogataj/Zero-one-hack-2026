"""Anomaly detection strategies for Task 3."""
from __future__ import annotations

from dataclasses import dataclass

from generate_sequences import validate_sequence


@dataclass
class AnomalyResult:
    is_valid: int       # 1 or 0
    score: float        # in [0.0, 1.0]; P(valid)
    predicted_rule: str # rule id when invalid; "" when valid


def detect_oracle(steps: list[str]) -> AnomalyResult:
    """Use the rule oracle: any reported violation → invalid + first rule reported."""
    violations = validate_sequence(steps)
    if violations:
        return AnomalyResult(is_valid=0, score=0.0, predicted_rule=violations[0].rule)
    return AnomalyResult(is_valid=1, score=1.0, predicted_rule="")


import math
from typing import Iterable


def detect_perplexity(
    steps: list[str],
    family: str,
    tokenizer,         # infineon_baseline.tokenizer.Tokenizer
    ngram,             # infineon_baseline.ngram.NGram
    threshold: float,  # log-prob threshold; below → invalid
) -> AnomalyResult:
    _, ids = tokenizer.encode(family, steps)
    lp = ngram.log_prob(family, ids)
    is_valid = 1 if lp >= threshold else 0
    score = _sigmoid(lp - threshold)
    return AnomalyResult(is_valid=is_valid, score=score, predicted_rule="")


def _sigmoid(x: float) -> float:
    # Saturates gracefully for very negative/positive x without overflow.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def calibrate_threshold(
    positives: Iterable[list[str]],
    negatives: Iterable[list[str]],
    families_pos: Iterable[str],
    families_neg: Iterable[str],
    tokenizer,
    ngram,
) -> float:
    """Pick the log-prob threshold that maximizes F1 on (pos, neg).

    Computes log-prob for every input, then evaluates every midpoint between
    sorted unique log-probs as a candidate threshold.
    """
    pos_lps = [ngram.log_prob(f, tokenizer.encode(f, s)[1]) for s, f in zip(positives, families_pos)]
    neg_lps = [ngram.log_prob(f, tokenizer.encode(f, s)[1]) for s, f in zip(negatives, families_neg)]
    all_lps = sorted(set(pos_lps + neg_lps))
    if len(all_lps) < 2:
        return all_lps[0] if all_lps else 0.0
    candidates = [(a + b) / 2 for a, b in zip(all_lps, all_lps[1:])]
    best_f1, best_th = -1.0, candidates[0]
    for th in candidates:
        tp = sum(1 for lp in pos_lps if lp >= th)
        fp = sum(1 for lp in neg_lps if lp >= th)
        fn = sum(1 for lp in pos_lps if lp < th)
        if tp + fp == 0 or tp + fn == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return best_th


def detect_hybrid(
    steps: list[str],
    family: str,
    tokenizer,
    ngram,
    threshold: float,
) -> AnomalyResult:
    """Oracle determines is_valid + predicted_rule; perplexity provides the continuous score."""
    oracle = detect_oracle(steps)
    perp = detect_perplexity(steps, family, tokenizer, ngram, threshold)
    return AnomalyResult(
        is_valid=oracle.is_valid,
        score=perp.score,
        predicted_rule=oracle.predicted_rule,
    )
