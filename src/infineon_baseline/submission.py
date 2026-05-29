"""Spec-compliant submission CSV writers with row validation."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from infineon_baseline.predictor import Task1Row, Task2Row, Task3Row


def _ensure_parent(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_task1_csv(rows: Iterable[Task1Row], path: Path) -> None:
    path = _ensure_parent(path)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])
        for r in rows:
            if len(r.ranks) != 5:
                raise ValueError(
                    f"Task1Row {r.example_id}: must emit exactly 5 ranks, got {len(r.ranks)}"
                )
            w.writerow([r.example_id, *r.ranks])


def write_task2_csv(rows: Iterable[Task2Row], path: Path) -> None:
    path = _ensure_parent(path)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "PREDICTED_SEQUENCE"])
        for r in rows:
            if "||" in r.predicted_sequence or r.predicted_sequence.startswith("|") or r.predicted_sequence.endswith("|"):
                raise ValueError(
                    f"Task2Row {r.example_id}: malformed pipe-separated sequence"
                )
            w.writerow([r.example_id, r.predicted_sequence])


def write_task3_csv(rows: Iterable[Task3Row], path: Path) -> None:
    path = _ensure_parent(path)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"])
        for r in rows:
            if r.is_valid not in (0, 1):
                raise ValueError(f"Task3Row {r.example_id}: IS_VALID must be 0 or 1, got {r.is_valid}")
            if not (0.0 <= r.score <= 1.0):
                raise ValueError(f"Task3Row {r.example_id}: SCORE must be in [0,1], got {r.score}")
            if r.is_valid == 1 and r.predicted_rule:
                raise ValueError(f"Task3Row {r.example_id}: PREDICTED_RULE must be empty when IS_VALID=1")
            w.writerow([r.example_id, r.is_valid, f"{r.score:.6f}", r.predicted_rule])


def write_meta(
    path: Path,
    task: str,
    model_info: dict,
    seed: int,
    eval_split_hash: str,
) -> None:
    path = _ensure_parent(path)
    path.write_text(json.dumps({
        "task": task,
        "model": model_info,
        "seed": seed,
        "eval_split_hash": eval_split_hash,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
