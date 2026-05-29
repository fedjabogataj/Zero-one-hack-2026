from infineon_baseline.eval_set import Cut, split, truncate
from tests.fixtures.mini_sequences import MINI_DATASET


def test_split_partitions_by_family_with_exact_holdout_size():
    train, hold = split(MINI_DATASET, holdout_per_family=3, seed=42)
    for family in MINI_DATASET:
        assert len(hold[family]) == 3
        assert len(train[family]) == len(MINI_DATASET[family]) - 3


def test_split_is_deterministic_for_same_seed():
    a = split(MINI_DATASET, holdout_per_family=3, seed=42)
    b = split(MINI_DATASET, holdout_per_family=3, seed=42)
    assert set(a[1]["mosfet"].keys()) == set(b[1]["mosfet"].keys())


def test_split_differs_for_different_seeds():
    a = split(MINI_DATASET, holdout_per_family=3, seed=1)
    b = split(MINI_DATASET, holdout_per_family=3, seed=2)
    assert set(a[1]["mosfet"].keys()) != set(b[1]["mosfet"].keys())


def test_split_raises_when_holdout_exceeds_corpus():
    import pytest
    with pytest.raises(ValueError, match="holdout_per_family"):
        split(MINI_DATASET, holdout_per_family=100, seed=42)


def test_truncate_at_60_percent_splits_partial_and_ground_truth():
    seq = list(range(100))  # 100 items
    cut = truncate(seq, fraction=0.6)
    assert isinstance(cut, Cut)
    assert cut.partial == list(range(60))
    assert cut.ground_truth == list(range(60, 100))
    assert cut.cut_index == 60


def test_truncate_uses_floor_for_non_integer_cut():
    seq = list(range(101))
    cut = truncate(seq, fraction=0.6)  # 60.6 -> floor 60
    assert cut.cut_index == 60
    assert len(cut.partial) + len(cut.ground_truth) == len(seq)


def test_truncate_rejects_invalid_fraction():
    import pytest
    for bad in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError, match="fraction must be in"):
            truncate([1, 2, 3], fraction=bad)


import pytest
from random import Random

from infineon_baseline import validate_sequence
from infineon_baseline.eval_set import inject_violation, SUPPORTED_RULES


# A long, fully-valid MOSFET sequence (from the reference data) used as injection substrate.
def _load_reference_mosfet() -> list[str]:
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "training_data" / "synthetic_mosfet.csv"
    rows = path.read_text().splitlines()[1:]  # skip header
    return [r.strip().strip('"') for r in rows if r.strip()]


@pytest.mark.parametrize("rule_id", SUPPORTED_RULES)
def test_injector_triggers_target_rule(rule_id):
    base = _load_reference_mosfet()
    assert validate_sequence(base) == [], "reference must start valid"
    rng = Random(42)
    corrupted, applied = inject_violation(base, rule_id, rng=rng)
    violations = validate_sequence(corrupted)
    assert any(v.rule == rule_id for v in violations), (
        f"expected {rule_id} in violations; got {[v.rule for v in violations]}"
    )
    assert applied == rule_id


def test_inject_violation_rejects_unknown_rule():
    with pytest.raises(ValueError, match="unknown rule"):
        inject_violation(_load_reference_mosfet(), "RULE_DOES_NOT_EXIST", rng=Random(0))


def test_inject_violation_returns_modified_copy_not_in_place():
    base = _load_reference_mosfet()
    snapshot = list(base)
    corrupted, _ = inject_violation(base, "RULE_DEP_NO_CLEAN", rng=Random(0))
    assert base == snapshot, "input must not be mutated"
    assert corrupted is not base


import csv

from infineon_baseline.eval_set import (
    build_valid_eval_inputs,
    build_anomaly_eval_inputs,
    write_eval_inputs_valid,
    write_eval_inputs_anomaly,
    write_ground_truth_valid,
    write_ground_truth_anomaly,
)


def test_build_valid_eval_inputs_has_correct_row_count():
    _, hold = split(MINI_DATASET, holdout_per_family=4, seed=42)
    rows = build_valid_eval_inputs(hold, seqs_per_family=4, fractions=(0.6, 0.8))
    # 4 families × seqs × 2 cuts -- here 3 families × 4 × 2 = 24
    assert len(rows) == 3 * 4 * 2


def test_build_valid_row_format():
    _, hold = split(MINI_DATASET, holdout_per_family=2, seed=42)
    rows = build_valid_eval_inputs(hold, seqs_per_family=2, fractions=(0.6,))
    r = rows[0]
    assert {"EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE"} <= r.keys()
    assert "|" in r["PARTIAL_SEQUENCE"]


def test_build_anomaly_eval_inputs_invalid_ratio_is_close_to_target():
    _, hold = split(MINI_DATASET, holdout_per_family=8, seed=42)
    rows = build_anomaly_eval_inputs(hold, seqs_per_family=8, invalid_ratio=0.5, seed=42)
    n_invalid = sum(1 for r in rows if r["_is_invalid"])
    # 8 × 3 = 24 total, invalid_ratio 0.5 → ~12.
    assert abs(n_invalid - 12) <= 3


def test_write_eval_inputs_valid_csv_round_trip(tmp_path):
    _, hold = split(MINI_DATASET, holdout_per_family=2, seed=42)
    rows = build_valid_eval_inputs(hold, seqs_per_family=2, fractions=(0.6, 0.8))
    out = tmp_path / "eval_input_valid.csv"
    write_eval_inputs_valid(rows, out)
    with out.open(newline="") as f:
        readback = list(csv.DictReader(f))
    assert readback[0]["FAMILY"] in {"mosfet", "igbt", "ic"}
    assert "|" in readback[0]["PARTIAL_SEQUENCE"]


def test_write_ground_truth_anomaly_columns(tmp_path):
    _, hold = split(MINI_DATASET, holdout_per_family=2, seed=42)
    rows = build_anomaly_eval_inputs(hold, seqs_per_family=2, invalid_ratio=0.5, seed=42)
    out = tmp_path / "gt_anomaly.csv"
    write_ground_truth_anomaly(rows, out)
    with out.open(newline="") as f:
        readback = list(csv.DictReader(f))
    assert set(readback[0].keys()) == {"EXAMPLE_ID", "IS_VALID", "RULE_VIOLATED"}
