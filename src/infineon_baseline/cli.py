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
    # Validate mutual exclusivity of --model and --transformer.
    has_model = bool(getattr(args, "model", None))
    has_transformer = bool(getattr(args, "transformer", None))
    if has_model and has_transformer:
        print("ERROR: --model and --transformer are mutually exclusive", file=sys.stderr)
        return 2
    if not has_model and not has_transformer:
        print("ERROR: one of --model or --transformer is required", file=sys.stderr)
        return 2

    wandb_run_id: str | None = None
    if has_transformer:
        from infineon_baseline.transformer_predictor import TransformerPredictor
        model = TransformerPredictor.load(Path(args.transformer))
        tokenizer = model._tokenizer
        model_kind = "transformer"
        # Recover the training run id (saved by train.py into the checkpoint) so
        # the downstream `score` step can resume the same wandb run.
        try:
            import torch
            ckpt = torch.load(Path(args.transformer), map_location="cpu", weights_only=False)
            wandb_run_id = ckpt.get("wandb_run_id")
        except Exception:
            pass
    else:
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

    model_order = getattr(model, "order", None)
    if has_transformer:
        model_info = {"type": "transformer", "path": str(args.transformer),
                      "wandb_run_id": wandb_run_id}
    else:
        model_info = {"type": model_kind, "order": model_order,
                      "embeddings": str(args.embeddings) if args.embeddings else None,
                      "wandb_run_id": None}
    write_meta(
        out_path.with_suffix(".meta.json"),
        task=args.task,
        model_info=model_info,
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
# Subcommand: build-st-embeddings
# --------------------------------------------------------------------------- #
def cmd_build_st_embeddings(args: argparse.Namespace) -> int:
    """Compute sentence-transformer embeddings for every step."""
    from infineon_baseline.embeddings_st import STStepEmbedder
    descriptions_dir = Path(args.descriptions_dir)
    paths = sorted(descriptions_dir.glob("*longdescription_parameters.csv"))
    if not paths:
        raise FileNotFoundError(f"no *longdescription_parameters.csv in {descriptions_dir}")
    embedder = STStepEmbedder.from_description_csvs(paths)
    out_path = Path(args.out)
    embedder.save(out_path)
    print(f"✓ ST-embedded {len(embedder.idx_to_step)} unique steps "
          f"({embedder.vectors.shape[1]} dims) → {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: train
# --------------------------------------------------------------------------- #
def cmd_train(args: argparse.Namespace) -> int:
    """Train the transformer LM."""
    from infineon_baseline.train import train as _train
    return _train(args)


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

    # ── Optional wandb logging ──────────────────────────────────────────── #
    _log_score_to_wandb(args, rep)
    return 0


def _log_score_to_wandb(args: argparse.Namespace, rep: dict) -> None:
    """Send the metrics report to wandb, optionally resuming a training run.

    Activates only if --wandb-project (or WANDB_PROJECT env var) is set.
    If --wandb-run-id is given, resumes that exact run so eval metrics land
    on the same chart as the training they correspond to. Otherwise creates
    a fresh "score" run linked to the same project. Silently no-ops if
    wandb is missing or init fails.
    """
    import os
    project = getattr(args, "wandb_project", None) or os.environ.get("WANDB_PROJECT")
    if not project:
        return
    try:
        import wandb
    except ImportError:
        print("⚠ wandb not installed (pip install 'infineon-baseline[tracking]'); "
              "skipping eval logging")
        return
    entity = (
        getattr(args, "wandb_entity", None)
        or os.environ.get("WANDB_ENTITY")
        or "fedja-bogataj-org"   # default account for this repo
    )
    run_id = getattr(args, "wandb_run_id", None)
    # Auto-detect: if --wandb-run-id wasn't passed, look for a sibling
    # `<predictions>.meta.json` written by `predict`. If it carries a
    # wandb_run_id, resume that run so eval metrics land on the same chart
    # as the training they correspond to.
    if not run_id:
        meta_path = Path(args.predictions).with_suffix(".meta.json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                candidate = (meta.get("model") or {}).get("wandb_run_id")
                if candidate:
                    run_id = candidate
                    print(f"📊 wandb: resuming run {run_id} (from {meta_path.name})")
            except Exception:
                pass
    init_kwargs: dict = {
        "project": project,
        "entity": entity,
        "mode": (getattr(args, "wandb_mode", None) or os.environ.get("WANDB_MODE")
                 or "online"),
        "settings": wandb.Settings(start_method="thread"),
    }
    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "must"
    else:
        init_kwargs["name"] = (getattr(args, "wandb_run_name", None)
                               or f"score-{args.task}")
        init_kwargs["job_type"] = "score"

    try:
        run = wandb.init(**init_kwargs)
    except Exception as e:
        print(f"⚠ wandb.init failed ({e}); skipping eval logging")
        return

    # Flatten the report dict into wandb-friendly key paths.
    payload: dict = {}
    for k, v in rep.get("overall", {}).items():
        if isinstance(v, (int, float)):
            payload[f"eval/{args.task}/overall/{k}"] = v
        elif isinstance(v, dict):  # e.g. confusion matrix for anomaly
            for ck, cv in v.items():
                if isinstance(cv, (int, float)):
                    payload[f"eval/{args.task}/overall/{k}/{ck}"] = cv
    for fam, fam_metrics in rep.get("per_family", {}).items():
        for k, v in fam_metrics.items():
            if isinstance(v, (int, float)):
                payload[f"eval/{args.task}/{fam}/{k}"] = v
            elif isinstance(v, dict):
                for ck, cv in v.items():
                    if isinstance(cv, (int, float)):
                        payload[f"eval/{args.task}/{fam}/{k}/{ck}"] = cv
    # Headline metrics get a flat alias so wandb's summary panel shows them.
    headline_keys = {
        "next-step": ("top_1_accuracy", "top_3_accuracy", "top_5_accuracy", "mrr"),
        "complete":  ("exact_match_rate", "normalized_edit_distance",
                      "token_accuracy", "block_accuracy"),
        "anomaly":   ("binary_accuracy", "precision", "recall", "f1", "roc_auc",
                      "rule_attribution_accuracy"),
    }.get(args.task, ())
    for k in headline_keys:
        v = rep.get("overall", {}).get(k)
        if isinstance(v, (int, float)):
            payload[f"summary/{args.task}/{k}"] = v

    try:
        wandb.log(payload)
        print(f"📊 wandb: logged {len(payload)} eval metrics → {run.url or '(offline)'}")
    except Exception as e:
        print(f"⚠ wandb.log failed ({e})")
    finally:
        try:
            wandb.finish()
        except Exception:
            pass


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
    p.add_argument("--model", default=None,
                   help="path to a fitted NGram/SoftNGram model (.pkl); "
                        "mutually exclusive with --transformer")
    p.add_argument("--transformer", default=None,
                   help="path to a trained TransformerPredictor checkpoint (.pt); "
                        "mutually exclusive with --model")
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

    p = sub.add_parser("build-st-embeddings",
                       help="compute sentence-transformer embeddings for every step")
    p.add_argument("--descriptions-dir", required=True,
                   help="dir containing *_longdescription_parameters.csv files")
    p.add_argument("--out", required=True, help="output .pkl path")

    p = sub.add_parser("train", help="train the transformer LM")
    p.add_argument("--train", required=True, help="path to train_split.csv")
    p.add_argument("--embeddings", required=True,
                   help="path to a fitted STStepEmbedder (.pkl)")
    p.add_argument("--out", required=True, help="output checkpoint path (.pt)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d-model", type=int, default=None, dest="d_model")
    p.add_argument("--n-heads", type=int, default=None, dest="n_heads")
    p.add_argument("--n-layers", type=int, default=None, dest="n_layers")
    # Weights & Biases experiment tracking (opt-in — only activates if --wandb-project
    # is passed or WANDB_PROJECT env var is set; falls back to no-op if wandb is
    # missing or login fails so it never blocks training).
    p.add_argument("--wandb-project", default=None,
                   help="W&B project name; if set, enables experiment tracking "
                        "(falls back to WANDB_PROJECT env var)")
    p.add_argument("--wandb-entity", default=None,
                   help="W&B entity / team (falls back to WANDB_ENTITY env var)")
    p.add_argument("--wandb-run-name", default=None,
                   help="W&B run name; defaults to auto-generated")
    p.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"],
                   help="W&B mode override; defaults to env var WANDB_MODE or 'online'")

    p = sub.add_parser("score", help="score predictions against ground truth")
    p.add_argument("--predictions", required=True)
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--task", required=True, choices=["next-step", "complete", "anomaly"])
    p.add_argument("--report-json", default=None)
    # Weights & Biases (opt-in — activate by passing --wandb-project or setting
    # WANDB_PROJECT in the environment). If --wandb-run-id is given, eval
    # metrics are appended to that training run; otherwise a fresh "score" run
    # is created. Default entity is fedja-bogataj-org (this repo's account).
    p.add_argument("--wandb-project", default=None,
                   help="W&B project; if set, logs eval metrics to wandb")
    p.add_argument("--wandb-entity", default=None,
                   help="W&B entity (default: fedja-bogataj-org)")
    p.add_argument("--wandb-run-id", default=None,
                   help="W&B run id to RESUME — eval metrics land on the same run as training")
    p.add_argument("--wandb-run-name", default=None,
                   help="W&B run name (only used if --wandb-run-id is not given)")
    p.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"],
                   help="W&B mode override (default: WANDB_MODE env var or 'online')")

    args = parser.parse_args(argv)
    dispatch = {
        "build-eval": cmd_build_eval,
        "fit": cmd_fit,
        "predict": cmd_predict,
        "score": cmd_score,
        "build-embeddings": cmd_build_embeddings,
        "build-st-embeddings": cmd_build_st_embeddings,
        "train": cmd_train,
    }
    return dispatch[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
