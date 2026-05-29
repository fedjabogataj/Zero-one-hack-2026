import json
from pathlib import Path

import pytest

from infineon_baseline.tokenizer import Tokenizer
from tests.fixtures.mini_sequences import MINI_DATASET, MINI_VOCAB


def test_fit_builds_sorted_deterministic_vocab():
    tok = Tokenizer.fit(MINI_DATASET)
    assert tok.id_to_step == sorted(MINI_VOCAB)
    # All step_to_id values are unique, 0..N-1
    assert sorted(tok.step_to_id.values()) == list(range(len(MINI_VOCAB)))
    # Families sorted alphabetically
    assert tok.family_to_id == {"ic": 0, "igbt": 1, "mosfet": 2}


def test_encode_decode_round_trip():
    tok = Tokenizer.fit(MINI_DATASET)
    steps = ["RECEIVE WAFER LOT", "LOT IDENTIFICATION", "SHIP LOT"]
    family_id, ids = tok.encode("mosfet", steps)
    assert family_id == tok.family_to_id["mosfet"]
    assert tok.decode(ids) == steps


def test_encode_unknown_step_raises():
    tok = Tokenizer.fit(MINI_DATASET)
    with pytest.raises(KeyError):
        tok.encode("mosfet", ["NOT A REAL STEP"])


def test_encode_unknown_family_raises():
    tok = Tokenizer.fit(MINI_DATASET)
    with pytest.raises(KeyError):
        tok.encode("nope", ["RECEIVE WAFER LOT"])


def test_save_and_load_round_trip(tmp_path):
    tok = Tokenizer.fit(MINI_DATASET)
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.step_to_id == tok.step_to_id
    assert loaded.id_to_step == tok.id_to_step
    assert loaded.family_to_id == tok.family_to_id


def test_two_fits_on_same_data_produce_identical_tokenizers():
    a = Tokenizer.fit(MINI_DATASET)
    b = Tokenizer.fit(MINI_DATASET)
    assert a.step_to_id == b.step_to_id
    assert a.family_to_id == b.family_to_id
