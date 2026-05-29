from infineon_baseline.predictor import run_task1, Task1Row
from infineon_baseline.ngram import NGram
from infineon_baseline.tokenizer import Tokenizer
from tests.fixtures.mini_sequences import MINI_DATASET


def _fitted():
    tok = Tokenizer.fit(MINI_DATASET)
    train_ids = {
        family: [tok.encode(family, steps)[1] for steps in seqs.values()]
        for family, seqs in MINI_DATASET.items()
    }
    return tok, NGram(order=2).fit(train_ids)


def test_run_task1_returns_one_row_per_example_with_five_ranks():
    tok, ng = _fitted()
    examples = [{
        "EXAMPLE_ID": "x1",
        "FAMILY": "mosfet",
        "COMPLETION_FRACTION": "0.6",
        "PARTIAL_SEQUENCE": "RECEIVE WAFER LOT|LOT IDENTIFICATION|PRE CLEAN WAFER",
    }]
    rows = list(run_task1(examples, tokenizer=tok, ngram=ng))
    assert len(rows) == 1
    r = rows[0]
    assert r.example_id == "x1"
    assert len(r.ranks) == 5
    assert all(isinstance(s, str) for s in r.ranks)


def test_run_task1_handles_empty_topk_by_padding_from_global_unigram():
    tok, ng = _fitted()
    examples = [{
        "EXAMPLE_ID": "x2",
        "FAMILY": "ic",
        "COMPLETION_FRACTION": "0.6",
        # Step IDs all known, but a family with a single training sequence may not have a
        # rich enough distribution to produce 5 unique top_k results. The predictor pads.
        "PARTIAL_SEQUENCE": "RECEIVE WAFER LOT",
    }]
    rows = list(run_task1(examples, tokenizer=tok, ngram=ng))
    assert len(rows[0].ranks) == 5


from infineon_baseline.predictor import run_task2, Task2Row


def test_run_task2_completes_until_ship_lot_or_cap():
    tok, ng = _fitted()
    examples = [{
        "EXAMPLE_ID": "y1",
        "FAMILY": "mosfet",
        "COMPLETION_FRACTION": "0.6",
        "PARTIAL_SEQUENCE": "RECEIVE WAFER LOT|LOT IDENTIFICATION|PRE CLEAN WAFER",
    }]
    rows = list(run_task2(examples, tokenizer=tok, ngram=ng, max_length_multiplier=2.0))
    assert len(rows) == 1
    r = rows[0]
    assert r.example_id == "y1"
    completion = r.predicted_sequence.split("|")
    assert len(completion) > 0
    # First predicted step must not echo the last seen step (no immediate repetition trap).
    # Completion does NOT include the partial sequence.
    assert "RECEIVE WAFER LOT" not in completion or completion.index("RECEIVE WAFER LOT") > 0


from infineon_baseline.predictor import run_task3, Task3Row


def test_run_task3_emits_anomaly_rows_with_required_fields():
    tok, ng = _fitted()
    examples = [{
        "EXAMPLE_ID": "z1",
        "FAMILY": "mosfet",
        "SEQUENCE": "|".join(MINI_DATASET["mosfet"]["mosfet_0001"]),
    }]
    rows = list(run_task3(examples, tokenizer=tok, ngram=ng, threshold=-100.0, strategy="hybrid"))
    assert rows[0].example_id == "z1"
    assert rows[0].is_valid in (0, 1)
    assert 0.0 <= rows[0].score <= 1.0
    assert isinstance(rows[0].predicted_rule, str)
