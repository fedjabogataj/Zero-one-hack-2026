import csv
from pathlib import Path

import pytest

from infineon_baseline.loaders import load_variants


def _write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["SEQUENCE_ID", "STEP"])
        for r in rows:
            w.writerow(r)


def test_load_variants_groups_by_sequence_id(tmp_path):
    csv_path = tmp_path / "MOSFET_variants.csv"
    _write_csv(csv_path, [
        ("seq_0001", "RECEIVE WAFER LOT"),
        ("seq_0001", "LOT IDENTIFICATION"),
        ("seq_0002", "RECEIVE WAFER LOT"),
        ("seq_0002", "SHIP LOT"),
    ])
    result = load_variants(csv_path)
    assert result == {
        "seq_0001": ["RECEIVE WAFER LOT", "LOT IDENTIFICATION"],
        "seq_0002": ["RECEIVE WAFER LOT", "SHIP LOT"],
    }


def test_load_variants_preserves_row_order(tmp_path):
    csv_path = tmp_path / "v.csv"
    _write_csv(csv_path, [
        ("s", "STEP_A"),
        ("s", "STEP_B"),
        ("s", "STEP_C"),
    ])
    assert load_variants(csv_path)["s"] == ["STEP_A", "STEP_B", "STEP_C"]


def test_load_variants_rejects_missing_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("WRONG_COL,OTHER\nx,y\n")
    with pytest.raises(ValueError, match="missing required column"):
        load_variants(csv_path)
