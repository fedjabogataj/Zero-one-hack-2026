import csv
from pathlib import Path

import pytest

from infineon_baseline.predictor import Task1Row, Task2Row, Task3Row
from infineon_baseline.submission import (
    write_task1_csv, write_task2_csv, write_task3_csv,
)


def _read(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_task1_writer_emits_exact_columns(tmp_path):
    out = tmp_path / "t1.csv"
    rows = [Task1Row(example_id="x", ranks=["A", "B", "C", "D", "E"])]
    write_task1_csv(rows, out)
    data = _read(out)
    assert list(data[0].keys()) == ["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"]
    assert data[0]["RANK_3"] == "C"


def test_task1_writer_refuses_wrong_rank_count(tmp_path):
    out = tmp_path / "t1.csv"
    rows = [Task1Row(example_id="x", ranks=["A", "B"])]
    with pytest.raises(ValueError, match="exactly 5 ranks"):
        write_task1_csv(rows, out)


def test_task2_writer_pipe_separated(tmp_path):
    out = tmp_path / "t2.csv"
    rows = [Task2Row(example_id="y", predicted_sequence="STEP_A|STEP_B|STEP_C")]
    write_task2_csv(rows, out)
    data = _read(out)
    assert data[0]["PREDICTED_SEQUENCE"] == "STEP_A|STEP_B|STEP_C"


def test_task3_writer_columns_and_value_ranges(tmp_path):
    out = tmp_path / "t3.csv"
    rows = [
        Task3Row(example_id="a", is_valid=1, score=0.95, predicted_rule=""),
        Task3Row(example_id="b", is_valid=0, score=0.08, predicted_rule="RULE_DEP_NO_CLEAN"),
    ]
    write_task3_csv(rows, out)
    data = _read(out)
    assert set(data[0].keys()) == {"EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"}
    assert data[0]["IS_VALID"] == "1" and data[0]["PREDICTED_RULE"] == ""
    assert data[1]["PREDICTED_RULE"] == "RULE_DEP_NO_CLEAN"


def test_task3_writer_rejects_invalid_is_valid(tmp_path):
    out = tmp_path / "t3.csv"
    rows = [Task3Row(example_id="a", is_valid=2, score=0.5, predicted_rule="")]
    with pytest.raises(ValueError, match="IS_VALID must be 0 or 1"):
        write_task3_csv(rows, out)


def test_task3_writer_rejects_score_out_of_range(tmp_path):
    out = tmp_path / "t3.csv"
    rows = [Task3Row(example_id="a", is_valid=1, score=1.5, predicted_rule="")]
    with pytest.raises(ValueError, match="SCORE must be in"):
        write_task3_csv(rows, out)
