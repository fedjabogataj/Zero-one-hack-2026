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
