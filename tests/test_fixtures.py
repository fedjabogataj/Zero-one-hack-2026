from tests.fixtures.mini_sequences import MINI_DATASET, MINI_VOCAB, FAMILIES

def test_mini_dataset_has_three_families():
    assert set(MINI_DATASET.keys()) == set(FAMILIES) == {"mosfet", "igbt", "ic"}

def test_each_family_has_at_least_ten_sequences():
    for family in FAMILIES:
        assert len(MINI_DATASET[family]) >= 10

def test_all_sequences_start_with_receive_and_end_with_ship():
    for family in FAMILIES:
        for seq_id, steps in MINI_DATASET[family].items():
            assert steps[0] == "RECEIVE WAFER LOT", f"{family}/{seq_id}"
            assert steps[-1] == "SHIP LOT", f"{family}/{seq_id}"

def test_vocab_is_union_of_all_steps():
    seen = set()
    for family in FAMILIES:
        for steps in MINI_DATASET[family].values():
            seen.update(steps)
    assert seen == set(MINI_VOCAB)
