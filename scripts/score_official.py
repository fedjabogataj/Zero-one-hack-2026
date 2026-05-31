#!/usr/bin/env python3
"""Score all model submissions using the organizer's official scorer.

Why this exists:
    Our internal scorer (src/infineon_baseline/metrics.py) was written
    against the published spec before the organizers distributed their
    eval_metrics.py. Two implementations of the same spec almost always
    drift on edge cases (tie-breaking in MRR, division-by-zero, etc.),
    so the model ranking under our scorer may not match the organizer's.

    This script runs official_eval/eval_metrics.py — which IS what the
    organizers will use to grade — against every model's existing
    submission CSVs (already produced by evaluate_all.sh). Output is
    a comparison table built from the organizer scorer's numbers, plus
    per-model JSON files so the results are queryable.

What it doesn't do:
    Re-predict. Predictions in outputs/submissions/<tag>/task{1,2,3}.csv
    are reused as-is. This is fast (~10s per model) since it skips the
    GPU step. If predictions are stale (model retrained without re-eval),
    run evaluate_all.sbatch first.

Schema translation:
    The organizer's scorer expects different GT column names than we
    produce. We translate once into outputs/eval_canonical_official_schema/
    and cache the result:

      OUR ground_truth_valid.csv             →  gt_next_step.csv
        EXAMPLE_ID, FULL_SEQUENCE, CUT_INDEX    EXAMPLE_ID, NEXT_STEP, FAMILY, COMPLETION_FRACTION
                                              →  gt_completion.csv
                                                 EXAMPLE_ID, PARTIAL_SEQUENCE, FULL_SEQUENCE, FAMILY, COMPLETION_FRACTION

      OUR ground_truth_anomaly.csv           →  gt_forbidden.csv   (rows with IS_VALID=0)
        EXAMPLE_ID, IS_VALID, RULE_VIOLATED     EXAMPLE_ID, VIOLATION_RULE
                                              →  gt_valid_supplement.csv  (rows with IS_VALID=1)
                                                 EXAMPLE_ID

Usage:
    python scripts/score_official.py                       # auto-discover models with submissions
    python scripts/score_official.py transformer_5000_bge-r0  transformer_5000_bge-r8
    python scripts/score_official.py --force               # ignore cached results

Output:
    outputs/eval_canonical_official_schema/gt_*.csv        # translated GT (cached)
    outputs/reports/<tag>/official_task_*.txt              # raw scorer stdout
    outputs/reports/<tag>/official.json                    # parsed numbers
    Plus a comparison table printed at the end.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_METRICS = PROJECT_ROOT / "official_eval" / "eval_metrics.py"
CANONICAL = PROJECT_ROOT / "outputs" / "eval_canonical"
TRANSLATED = PROJECT_ROOT / "outputs" / "eval_canonical_official_schema"
SUBMISSIONS = PROJECT_ROOT / "outputs" / "submissions"
REPORTS = PROJECT_ROOT / "outputs" / "reports"


# ─── GT schema translation ────────────────────────────────────────────────── #

def translate_gt(force: bool = False) -> dict[str, Path]:
    """Convert our GT files into the schema eval_metrics.py expects.

    Cached: skipped on subsequent runs unless --force. The translation is
    deterministic given the input GT files, so caching is safe.
    """
    TRANSLATED.mkdir(parents=True, exist_ok=True)
    next_step = TRANSLATED / "gt_next_step.csv"
    completion = TRANSLATED / "gt_completion.csv"
    forbidden = TRANSLATED / "gt_forbidden.csv"
    valid_supp = TRANSLATED / "gt_valid_supplement.csv"

    if not force and all(p.exists() for p in (next_step, completion, forbidden, valid_supp)):
        return {"next-step": next_step, "completion": completion,
                "anomaly": forbidden, "valid_supp": valid_supp}

    print(f"[translate_gt] writing translated GT → {TRANSLATED.relative_to(PROJECT_ROOT)}/")

    # ── Task 1 + 2: read eval_input + ground_truth_valid, join on EXAMPLE_ID ──
    eval_input = {}
    with (CANONICAL / "eval_input_valid.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            eval_input[r["EXAMPLE_ID"]] = r

    with (CANONICAL / "ground_truth_valid.csv").open(newline="") as f, \
         next_step.open("w", newline="") as f1, \
         completion.open("w", newline="") as f2:
        w1 = csv.writer(f1)
        w1.writerow(["EXAMPLE_ID", "NEXT_STEP", "FAMILY", "COMPLETION_FRACTION"])
        w2 = csv.writer(f2)
        w2.writerow(["EXAMPLE_ID", "PARTIAL_SEQUENCE", "FULL_SEQUENCE",
                     "FAMILY", "COMPLETION_FRACTION"])
        n = 0
        for r in csv.DictReader(f):
            eid = r["EXAMPLE_ID"]
            inp = eval_input.get(eid)
            if inp is None:
                continue
            full = r["FULL_SEQUENCE"]
            cut = int(r["CUT_INDEX"])
            steps = full.split("|")
            if not (0 <= cut < len(steps)):
                continue
            next_step_val = steps[cut]
            fam = inp["FAMILY"]
            frac = inp["COMPLETION_FRACTION"]
            partial = inp["PARTIAL_SEQUENCE"]
            w1.writerow([eid, next_step_val, fam, frac])
            w2.writerow([eid, partial, full, fam, frac])
            n += 1
        print(f"  next-step + completion: {n} rows")

    # ── Task 3: split mixed-label GT into two files ──────────────────────────
    with (CANONICAL / "ground_truth_anomaly.csv").open(newline="") as f, \
         forbidden.open("w", newline="") as ff, \
         valid_supp.open("w", newline="") as fv:
        wf = csv.writer(ff)
        wf.writerow(["EXAMPLE_ID", "VIOLATION_RULE"])
        wv = csv.writer(fv)
        wv.writerow(["EXAMPLE_ID"])
        n_forbidden = n_valid = 0
        for r in csv.DictReader(f):
            eid = r["EXAMPLE_ID"]
            iv = int(r["IS_VALID"])
            # Our column is RULE_VIOLATED; theirs is VIOLATION_RULE.
            rule = r.get("RULE_VIOLATED") or r.get("VIOLATION_RULE") or ""
            if iv == 0:
                wf.writerow([eid, rule])
                n_forbidden += 1
            else:
                wv.writerow([eid])
                n_valid += 1
        print(f"  anomaly: {n_forbidden} forbidden + {n_valid} valid")

    return {"next-step": next_step, "completion": completion,
            "anomaly": forbidden, "valid_supp": valid_supp}


# ─── Run + parse the organizer scorer ─────────────────────────────────────── #

# Each entry: metric_key -> regex that matches the scorer's printed line.
# The scorer prints under "EVAL RESULTS — …" headers; we only care about ALL.
_PATTERNS_TASK = {
    "next-step": {
        "top_1_accuracy": r"^Top-1 Accuracy\s*:\s*(\d+\.\d+)",
        "top_3_accuracy": r"^Top-3 Accuracy\s*:\s*(\d+\.\d+)",
        "top_5_accuracy": r"^Top-5 Accuracy\s*:\s*(\d+\.\d+)",
        "mrr":            r"^MRR\s*:\s*(\d+\.\d+)",
    },
    "completion": {
        "normalized_edit_distance": r"^Mean Normalized Edit Distance\s*:\s*(\d+\.\d+)",
        "exact_match_rate":         r"^Exact Match Rate\s*:\s*(\d+\.\d+)",
        "token_accuracy":           r"^Mean Token Accuracy\s*:\s*(\d+\.\d+)",
        "block_accuracy":           r"^Mean Block-level Accuracy\s*:\s*(\d+\.\d+)",
    },
    "anomaly": {
        "binary_accuracy":           r"^Binary Accuracy\s*:\s*(\d+\.\d+)",
        "precision":                 r"^Precision \(invalid class\)\s*:\s*(\d+\.\d+)",
        "recall":                    r"^Recall \(invalid class\)\s*:\s*(\d+\.\d+)",
        "f1":                        r"^F1 \(invalid class\)\s*:\s*(\d+\.\d+)",
        "roc_auc":                   r"^ROC-AUC\s*:\s*(\d+\.\d+)",
        "rule_attribution_accuracy": r"^Rule Attribution Accuracy\s*:\s*(\d+\.\d+)",
    },
}


def parse_metrics(stdout: str, task: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, pat in _PATTERNS_TASK[task].items():
        m = re.search(pat, stdout, re.MULTILINE)
        if m:
            out[key] = float(m.group(1))
    return out


def task_to_filename(task: str) -> str:
    # task1 = next-step, task2 = completion, task3 = anomaly
    return {"next-step": "task1.csv", "completion": "task2.csv", "anomaly": "task3.csv"}[task]


def score_one(tag: str, task: str, gt_paths: dict[str, Path]) -> tuple[str, dict[str, float]]:
    """Run the organizer scorer for ONE (model, task) pair. Returns (stdout, parsed)."""
    pred_path = SUBMISSIONS / tag / task_to_filename(task)
    if not pred_path.exists():
        return ("", {})
    cmd = [
        sys.executable, str(EVAL_METRICS),
        "--task", task,
        "--ground-truth", str(gt_paths[task]),
        "--predictions", str(pred_path),
    ]
    if task == "anomaly":
        cmd += ["--valid-supplement", str(gt_paths["valid_supp"])]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  ⚠ {task}: scorer exited {res.returncode}")
        if res.stderr:
            print(f"     stderr: {res.stderr.strip()[:200]}")
    return (res.stdout, parse_metrics(res.stdout, task))


# ─── Comparison table ─────────────────────────────────────────────────────── #

def print_table(all_results: dict[str, dict[str, dict[str, float]]]) -> None:
    """Print one section per task with all tags as columns."""
    tags = sorted(all_results)
    if not tags:
        print("(no models scored)")
        return
    direction = {  # ↑ = higher better, ↓ = lower better
        "top_1_accuracy": "↑", "top_3_accuracy": "↑", "top_5_accuracy": "↑", "mrr": "↑",
        "normalized_edit_distance": "↓", "exact_match_rate": "↑",
        "token_accuracy": "↑", "block_accuracy": "↑",
        "binary_accuracy": "↑", "precision": "↑", "recall": "↑", "f1": "↑",
        "roc_auc": "↑", "rule_attribution_accuracy": "↑",
    }

    for task in ("next-step", "completion", "anomaly"):
        metrics = list(_PATTERNS_TASK[task].keys())
        print()
        print("═" * 100)
        print(f"Task: {task}    (scored by official_eval/eval_metrics.py)")
        print("═" * 100)
        col_w = max(28, max(len(t) for t in tags) + 2)
        # Header
        header = f"{'metric':<32}" + "".join(f"{t:<{col_w}}" for t in tags)
        print(header)
        print("-" * len(header))
        # Body: one row per metric, find the best column
        for m in metrics:
            arrow = direction.get(m, "")
            vals = [all_results[t].get(task, {}).get(m) for t in tags]
            if all(v is None for v in vals):
                continue
            # Determine best: highest if ↑, lowest if ↓
            non_null = [v for v in vals if v is not None]
            if not non_null:
                best = None
            elif arrow == "↓":
                best = min(non_null)
            else:
                best = max(non_null)
            row = f"{m + ' ' + arrow:<32}"
            for v in vals:
                if v is None:
                    row += f"{'—':<{col_w}}"
                else:
                    s = f"*{v:.4f}*" if v == best else f"{v:.4f}"
                    row += f"{s:<{col_w}}"
            print(row)
    print()


# ─── Entry point ──────────────────────────────────────────────────────────── #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-score all model submissions with the organizer's scorer.")
    parser.add_argument("tags", nargs="*",
                        help="model tags to score (default: all dirs in outputs/submissions/)")
    parser.add_argument("--force", action="store_true",
                        help="ignore cached translated GT and cached results")
    args = parser.parse_args()

    if not EVAL_METRICS.exists():
        print(f"ERROR: {EVAL_METRICS} not found", file=sys.stderr)
        return 2
    if not CANONICAL.exists():
        print(f"ERROR: {CANONICAL} not found — run build-test-only first", file=sys.stderr)
        return 2

    gt_paths = translate_gt(force=args.force)

    if args.tags:
        tags = args.tags
    else:
        tags = sorted(
            p.name for p in SUBMISSIONS.iterdir()
            if p.is_dir() and (p / "task1.csv").exists()
                          and (p / "task2.csv").exists()
                          and (p / "task3.csv").exists()
        )
    if not tags:
        print("No models with full submission CSVs found in outputs/submissions/", file=sys.stderr)
        return 2

    print(f"Scoring {len(tags)} model(s) with the official scorer\n")

    all_results: dict[str, dict[str, dict[str, float]]] = {}
    for tag in tags:
        rep_dir = REPORTS / tag
        official_json = rep_dir / "official.json"
        if not args.force and official_json.exists():
            try:
                all_results[tag] = json.loads(official_json.read_text())
                print(f"⏭  {tag}: cached")
                continue
            except Exception:
                pass

        rep_dir.mkdir(parents=True, exist_ok=True)
        print(f"▶  {tag}")
        tag_results: dict[str, dict[str, float]] = {}
        for task in ("next-step", "completion", "anomaly"):
            stdout, parsed = score_one(tag, task, gt_paths)
            if not stdout:
                print(f"    ⚠ {task}: prediction CSV missing")
                continue
            (rep_dir / f"official_{task}.txt").write_text(stdout)
            tag_results[task] = parsed
            print(f"    ✓ {task}: " +
                  ", ".join(f"{k}={v:.4f}" for k, v in parsed.items()))
        official_json.write_text(json.dumps(tag_results, indent=2))
        all_results[tag] = tag_results

    print_table(all_results)
    print(f"Per-model JSONs:  outputs/reports/<tag>/official.json")
    print(f"Raw scorer logs:  outputs/reports/<tag>/official_<task>.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
