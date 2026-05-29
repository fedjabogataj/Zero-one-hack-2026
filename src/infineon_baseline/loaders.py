"""Variant-CSV loaders."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def load_variants(csv_path: Path) -> dict[str, list[str]]:
    """Read a long-format `*_variants.csv` into {sequence_id: [step, ...]}.

    Required columns: SEQUENCE_ID, STEP. Row order within a sequence is preserved.
    """
    csv_path = Path(csv_path)
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "SEQUENCE_ID" not in reader.fieldnames or "STEP" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path}: missing required column(s); "
                f"need SEQUENCE_ID and STEP, got {reader.fieldnames}"
            )
        out: dict[str, list[str]] = defaultdict(list)
        for row in reader:
            out[row["SEQUENCE_ID"]].append(row["STEP"])
    return dict(out)
