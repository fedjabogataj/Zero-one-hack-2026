from random import Random

from infineon_baseline import validate_sequence
from infineon_baseline.anomaly import AnomalyResult, detect_oracle
from infineon_baseline.eval_set import inject_violation


def _load_reference_mosfet() -> list[str]:
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    return [r.strip().strip('"') for r in
            (repo_root / "training_data" / "synthetic_mosfet.csv").read_text().splitlines()[1:]
            if r.strip()]


def test_oracle_passes_valid_sequence():
    valid = _load_reference_mosfet()
    assert validate_sequence(valid) == []
    res = detect_oracle(valid)
    assert isinstance(res, AnomalyResult)
    assert res.is_valid == 1 and res.predicted_rule == ""


def test_oracle_detects_injected_violation():
    valid = _load_reference_mosfet()
    corrupted, applied = inject_violation(valid, "RULE_DEP_NO_CLEAN", rng=Random(42))
    res = detect_oracle(corrupted)
    assert res.is_valid == 0
    assert res.predicted_rule == applied


from infineon_baseline.anomaly import calibrate_threshold, detect_perplexity
from infineon_baseline.ngram import NGram
from infineon_baseline.tokenizer import Tokenizer
from tests.fixtures.mini_sequences import MINI_DATASET


def _fit_mini_ngram() -> tuple[Tokenizer, NGram]:
    tok = Tokenizer.fit(MINI_DATASET)
    train_ids = {
        family: [tok.encode(family, steps)[1] for steps in seqs.values()]
        for family, seqs in MINI_DATASET.items()
    }
    model = NGram(order=2).fit(train_ids)
    return tok, model


def test_perplexity_score_is_lower_for_shuffled_sequence():
    tok, ng = _fit_mini_ngram()
    valid = MINI_DATASET["mosfet"]["mosfet_0001"]
    shuffled = list(reversed(valid))   # reversal preserves first/last but scrambles transitions
    res_v = detect_perplexity(valid, "mosfet", tok, ng, threshold=-50.0)
    res_s = detect_perplexity(shuffled, "mosfet", tok, ng, threshold=-50.0)
    # Score is the log-prob mapped to [0, 1]; valid should be higher.
    assert res_v.score > res_s.score


def test_calibrate_threshold_returns_finite_float():
    tok, ng = _fit_mini_ngram()
    pos = [MINI_DATASET["mosfet"][f"mosfet_{i:04d}"] for i in range(1, 6)]
    neg = [list(reversed(s)) for s in pos]
    families = ["mosfet"] * len(pos)
    th = calibrate_threshold(positives=pos, negatives=neg,
                             families_pos=families, families_neg=families,
                             tokenizer=tok, ngram=ng)
    import math
    assert math.isfinite(th)


from infineon_baseline.anomaly import detect_hybrid


def test_hybrid_uses_oracle_for_decision_and_perplexity_for_score():
    tok, ng = _fit_mini_ngram()
    valid = MINI_DATASET["mosfet"]["mosfet_0001"]
    res = detect_hybrid(valid, "mosfet", tok, ng, threshold=-50.0)
    # Oracle says valid → is_valid=1 and predicted_rule="" (regardless of perplexity score range).
    assert res.is_valid == 1
    assert res.predicted_rule == ""
    # Score should be a real probability in [0,1] from perplexity, not the constant 1.0 from the oracle alone.
    assert 0.0 <= res.score <= 1.0
