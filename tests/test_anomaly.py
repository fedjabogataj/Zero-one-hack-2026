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
