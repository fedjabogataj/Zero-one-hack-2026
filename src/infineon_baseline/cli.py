"""Single CLI entry point with four subcommands.

Usage:
    infineon-baseline build-eval --variants-dir DIR --out DIR --seed 42
    infineon-baseline fit        --train DIR --out PATH [--order 3]
    infineon-baseline predict    --model PATH --eval-input PATH --task TASK --out PATH
    infineon-baseline score      --predictions PATH --ground-truth PATH --task TASK
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from infineon_baseline.anomaly import calibrate_threshold
from infineon_baseline.embeddings import StepEmbedder
from infineon_baseline.eval_set import (
    build_anomaly_eval_inputs, build_valid_eval_inputs,
    split, write_eval_inputs_anomaly, write_eval_inputs_valid,
    write_ground_truth_anomaly, write_ground_truth_valid,
)
from infineon_baseline.loaders import load_variants
from infineon_baseline.metrics import report
from infineon_baseline.ngram import NGram
from infineon_baseline.predictor import run_task1, run_task2, run_task3
from infineon_baseline.soft_ngram import SoftNGram, build_step_embedder_from_training_data
from infineon_baseline.submission import (
    write_meta, write_task1_csv, write_task2_csv, write_task3_csv,
)
from infineon_baseline.tokenizer import Tokenizer


def _load_corpus(variants_dir: Path) -> dict[str, dict[str, list[str]]]:
    """Load *_variants.csv for all three families from a directory."""
    by_family: dict[str, dict[str, list[str]]] = {}
    for family in ("mosfet", "igbt", "ic"):
        candidates = list(Path(variants_dir).glob(f"*{family.upper()}*variants*.csv"))
        if not candidates:
            raise FileNotFoundError(f"no {family} variants CSV in {variants_dir}")
        by_family[family] = load_variants(candidates[0])
    return by_family


def _hash_split(holdout_ids: list[str]) -> str:
    h = hashlib.sha256()
    for sid in sorted(holdout_ids):
        h.update(sid.encode())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Subcommand: build-eval
# --------------------------------------------------------------------------- #
def cmd_build_eval(args: argparse.Namespace) -> int:
    corpus = _load_corpus(Path(args.variants_dir))
    train, hold = split(corpus, holdout_per_family=args.holdout_per_family, seed=args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Valid eval inputs (Tasks 1 & 2): use first half of hold-out per family for truncation.
    valid_rows = build_valid_eval_inputs(
        hold, seqs_per_family=args.holdout_per_family // 2, fractions=(0.6, 0.8),
    )
    write_eval_inputs_valid(valid_rows, out_dir / "eval_input_valid.csv")
    write_ground_truth_valid(valid_rows, out_dir / "ground_truth_valid.csv")

    # Anomaly eval inputs (Task 3): full hold-out, with target invalid ratio.
    anom_rows = build_anomaly_eval_inputs(
        hold,
        seqs_per_family=args.holdout_per_family,
        invalid_ratio=args.anomaly_invalid_ratio,
        seed=args.seed,
    )
    write_eval_inputs_anomaly(anom_rows, out_dir / "eval_input_anomaly.csv")
    write_ground_truth_anomaly(anom_rows, out_dir / "ground_truth_anomaly.csv")

    # Persist the training split for `fit`.
    train_path = out_dir / "train_split.csv"
    with train_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["SEQUENCE_ID", "FAMILY", "STEP"])
        for family in sorted(train):
            for sid in sorted(train[family]):
                for step in train[family][sid]:
                    w.writerow([sid, family, step])

    holdout_ids = [sid for fam_seqs in hold.values() for sid in fam_seqs]
    meta = {
        "seed": args.seed,
        "holdout_per_family": args.holdout_per_family,
        "anomaly_invalid_ratio": args.anomaly_invalid_ratio,
        "split_hash": _hash_split(holdout_ids),
        "n_train_seqs": sum(len(s) for s in train.values()),
        "n_hold_seqs": sum(len(s) for s in hold.values()),
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"✓ wrote eval files to {out_dir}/  (split_hash={meta['split_hash']})")
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: fit
# --------------------------------------------------------------------------- #
def cmd_fit(args: argparse.Namespace) -> int:
    # Read train_split.csv into {family: {sid: [steps]}}
    train_split: dict[str, dict[str, list[str]]] = {}
    with Path(args.train).open(newline="") as f:
        for row in csv.DictReader(f):
            train_split.setdefault(row["FAMILY"], {}).setdefault(row["SEQUENCE_ID"], []).append(row["STEP"])
    tokenizer = Tokenizer.fit(train_split)
    train_ids = {
        family: [tokenizer.encode(family, steps)[1] for steps in seqs.values()]
        for family, seqs in train_split.items()
    }
    model = NGram(order=args.order, mask_family_vocab=args.mask_family).fit(train_ids)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    tokenizer.save(out_path.with_suffix(".tokenizer.json"))
    print(f"✓ fitted n-gram (order={args.order}) over {sum(len(s) for s in train_split.values())} sequences")
    print(f"  → {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: predict
# --------------------------------------------------------------------------- #
def _read_eval_input(path: Path) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def cmd_predict(args: argparse.Namespace) -> int:
    base_ngram = NGram.load(args.model)
    tokenizer = Tokenizer.load(Path(args.model).with_suffix(".tokenizer.json"))

    # If embeddings are supplied, wrap the n-gram in a SoftNGram so unknown
    # step names get aliased to nearest known + unseen prefixes fall back to
    # embedding-similar prefixes' counters.
    if args.embeddings:
        embedder = StepEmbedder.load(args.embeddings)
        model = SoftNGram(ngram=base_ngram, embedder=embedder, tokenizer=tokenizer)
        model_kind = "soft-ngram"
    else:
        model = base_ngram
        model_kind = "ngram"

    examples = _read_eval_input(args.eval_input)
    out_path = Path(args.out)

    if args.task == "next-step":
        rows = list(run_task1(examples, tokenizer=tokenizer, ngram=model))
        write_task1_csv(rows, out_path)
    elif args.task == "complete":
        rows = list(run_task2(examples, tokenizer=tokenizer, ngram=model, constrain=args.constrain))
        write_task2_csv(rows, out_path)
    elif args.task == "anomaly":
        threshold = args.threshold if args.threshold is not None else 0.0
        rows = list(run_task3(examples, tokenizer=tokenizer, ngram=model,
                              threshold=threshold, strategy=args.anomaly_strategy))
        write_task3_csv(rows, out_path)
    else:
        print(f"unknown task: {args.task}", file=sys.stderr)
        return 2

    write_meta(
        out_path.with_suffix(".meta.json"),
        task=args.task,
        model_info={"type": model_kind, "order": base_ngram.order,
                    "embeddings": str(args.embeddings) if args.embeddings else None},
        seed=args.seed,
        eval_split_hash="(set externally)",
    )
    print(f"✓ wrote {len(rows)} predictions ({model_kind}) → {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: build-embeddings
# --------------------------------------------------------------------------- #
def cmd_build_embeddings(args: argparse.Namespace) -> int:
    """Compute TF-IDF embeddings for every step from the description CSVs."""
    descriptions_dir = Path(args.descriptions_dir)
    embedder = build_step_embedder_from_training_data(descriptions_dir)
    out_path = Path(args.out)
    embedder.save(out_path)
    print(f"✓ embedded {len(embedder.idx_to_step)} unique steps "
          f"({embedder.vectors.shape[1]} dims) → {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: score
# --------------------------------------------------------------------------- #
def cmd_score(args: argparse.Namespace) -> int:
    preds = pd.read_csv(args.predictions)
    gt = pd.read_csv(args.ground_truth)
    # Carry FAMILY across from preds if absent in gt, and vice versa (joins on EXAMPLE_ID).
    if "FAMILY" not in gt.columns and "FAMILY" in preds.columns:
        gt = gt.merge(preds[["EXAMPLE_ID", "FAMILY"]], on="EXAMPLE_ID", how="left")
    if "FAMILY" not in preds.columns and "FAMILY" in gt.columns:
        preds = preds.merge(gt[["EXAMPLE_ID", "FAMILY"]], on="EXAMPLE_ID", how="left")
    # For Task 2: gt column rename for our metrics' expected FULL/GROUND_TRUTH names.
    if args.task == "complete" and "FULL_SEQUENCE" in gt.columns and "GROUND_TRUTH_SEQUENCE" not in gt.columns:
        # The ground truth FULL_SEQUENCE is the entire sequence; we want only the post-cut suffix.
        gt["GROUND_TRUTH_SEQUENCE"] = gt.apply(
            lambda r: "|".join(r["FULL_SEQUENCE"].split("|")[int(r["CUT_INDEX"]):]),
            axis=1,
        )
    if args.task == "next-step" and "NEXT_STEP" not in gt.columns and "FULL_SEQUENCE" in gt.columns:
        gt["NEXT_STEP"] = gt.apply(
            lambda r: r["FULL_SEQUENCE"].split("|")[int(r["CUT_INDEX"])],
            axis=1,
        )
    rep = report(args.task, predictions=preds, ground_truth=gt)
    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(json.dumps(rep, indent=2, default=str))
        print(f"✓ wrote report → {args.report_json}")
    return 0


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="infineon-baseline")
    parser.add_argument("--seed", type=int, default=42, help="global random seed")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-eval", help="build held-out eval set + ground truth")
    p.add_argument("--variants-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--holdout-per-family", type=int, default=200)
    p.add_argument("--anomaly-invalid-ratio", type=float, default=0.39)

    p = sub.add_parser("fit", help="fit the n-gram on a train split")
    p.add_argument("--train", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--mask-family", action="store_true",
                   help="zero probabilities for steps unseen in that family")

    p = sub.add_parser("predict", help="produce a submission for one task")
    p.add_argument("--model", required=True)
    p.add_argument("--eval-input", required=True)
    p.add_argument("--task", required=True, choices=["next-step", "complete", "anomaly"])
    p.add_argument("--out", required=True)
    p.add_argument("--constrain", action="store_true", help="rule-aware decoder for complete")
    p.add_argument("--anomaly-strategy", choices=["oracle", "perplexity", "hybrid"], default="hybrid")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--embeddings", default=None,
                   help="path to a StepEmbedder pickle; if set, wraps the n-gram in a SoftNGram "
                        "(OOD-friendly: unknown step names alias to nearest known, unseen prefixes "
                        "fall back to embedding-similar prefixes)")

    p = sub.add_parser("build-embeddings", help="compute TF-IDF embeddings for every step")
    p.add_argument("--descriptions-dir", required=True,
                   help="dir containing *_longdescription_parameters.csv files")
    p.add_argument("--out", required=True, help="output .pkl path")

    p = sub.add_parser("score", help="score predictions against ground truth")
    p.add_argument("--predictions", required=True)
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--task", required=True, choices=["next-step", "complete", "anomaly"])
    p.add_argument("--report-json", default=None)

    args = parser.parse_args(argv)
    dispatch = {
        "build-eval": cmd_build_eval,
        "fit": cmd_fit,
        "predict": cmd_predict,
        "score": cmd_score,
        "build-embeddings": cmd_build_embeddings,
    }
    return dispatch[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
