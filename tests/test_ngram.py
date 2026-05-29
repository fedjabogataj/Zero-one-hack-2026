import pytest

from infineon_baseline.ngram import NGram


def _toy_corpus() -> dict[str, list[list[int]]]:
    # Two MOSFET sequences in id-space; vocab = {0..4}
    return {
        "mosfet": [
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 4],
        ],
        "igbt": [[0, 2, 4]],
    }


def test_top_k_after_seen_prefix():
    model = NGram(order=2).fit(_toy_corpus())
    # After "1, 2" we always see "3" twice; after "0,1,2,4" once. So top_k("mosfet", (1,2)) starts with 3.
    top = model.top_k("mosfet", (1, 2), k=2)
    assert top[0] == 3


def test_top_k_returns_at_most_k():
    model = NGram(order=3).fit(_toy_corpus())
    assert len(model.top_k("mosfet", (0, 1, 2), k=10)) <= 10  # bounded by vocab seen


def test_top_k_backoff_to_shorter_prefix_when_unseen():
    model = NGram(order=3).fit(_toy_corpus())
    # (99, 99, 99) never seen at order 3; should back off to order 2 then unigram.
    top = model.top_k("mosfet", (99, 99, 99), k=1)
    assert top != []  # unigram fallback always returns something


def test_top_k_for_unknown_family_returns_empty():
    model = NGram(order=2).fit(_toy_corpus())
    assert model.top_k("unknown", (0, 1), k=5) == []


def test_fit_returns_self_for_chaining():
    m = NGram(order=2)
    assert m.fit(_toy_corpus()) is m


import math


def test_log_prob_is_higher_for_seen_sequence():
    model = NGram(order=2).fit(_toy_corpus())
    seen = [0, 1, 2, 3, 4]
    unseen = [4, 3, 2, 1, 0]  # reverse — never seen as transitions
    assert model.log_prob("mosfet", seen) > model.log_prob("mosfet", unseen)


def test_log_prob_is_finite_for_completely_unseen_transitions():
    model = NGram(order=2).fit(_toy_corpus())
    # Sequence of unknown ids — stupid backoff falls to unigram, which uses a floor.
    val = model.log_prob("mosfet", [42, 43, 44])
    assert math.isfinite(val)


def test_log_prob_for_unknown_family_is_minus_inf():
    model = NGram(order=2).fit(_toy_corpus())
    assert model.log_prob("unknown_family", [0, 1, 2]) == float("-inf")


def test_save_and_load_round_trip(tmp_path):
    model = NGram(order=3).fit(_toy_corpus())
    path = tmp_path / "ngram.pkl"
    model.save(path)
    loaded = NGram.load(path)
    assert loaded.order == model.order
    assert loaded.top_k("mosfet", (1, 2), k=2) == model.top_k("mosfet", (1, 2), k=2)
    assert loaded.log_prob("mosfet", [0, 1, 2]) == model.log_prob("mosfet", [0, 1, 2])
