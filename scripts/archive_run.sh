#!/usr/bin/env bash
# Promote a finished experiment from outputs/ into results/ — copies the
# metrics JSON, submission CSVs + meta, optionally the model checkpoint,
# and writes a self-contained README.md with the headline numbers.
#
# Usage:
#   ./scripts/archive_run.sh <MODEL_PATH>                      # metrics + submissions only
#   ./scripts/archive_run.sh <MODEL_PATH> --include-model      # also copy the .pt/.pkl (large!)
#   ./scripts/archive_run.sh --all                             # archive every model under outputs/reports/
#   ./scripts/archive_run.sh --all --include-model
#
# Examples:
#   ./scripts/archive_run.sh outputs/models/transformer_5000.pt
#   ./scripts/archive_run.sh outputs/models/ngram_o3.pkl --include-model
#   ./scripts/archive_run.sh --all
#
# Output structure (per archived run):
#   results/<tag>/
#   ├── README.md             ← auto-generated summary with metrics table + config
#   ├── metrics/
#   │   ├── task1.json
#   │   ├── task2.json
#   │   └── task3.json
#   ├── submissions/
#   │   ├── task1.csv + task1.meta.json
#   │   ├── task2.csv + task2.meta.json
#   │   └── task3.csv + task3.meta.json
#   └── model.pt              ← only if --include-model

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Pick the python runner so we can format the README in Python (markdown table
# generation is painful in bash).
if command -v pixi >/dev/null 2>&1 && [[ -d .pixi || -f pixi.toml ]]; then
    PYTHON=(pixi run python)
elif [[ -d .venv ]] && [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PYTHON=(python)
else
    PYTHON=(python)
fi

# ─── Parse args ────────────────────────────────────────────────────────────
INCLUDE_MODEL=false
ALL_MODE=false
MODELS=()

for arg in "$@"; do
    case "$arg" in
        --include-model) INCLUDE_MODEL=true ;;
        --all)           ALL_MODE=true ;;
        --help|-h)
            sed -n '1,/^set -euo/p' "$0" | head -n -1
            exit 0
            ;;
        *) MODELS+=("$arg") ;;
    esac
done

if $ALL_MODE; then
    MODELS=()
    for d in outputs/reports/*/; do
        tag=$(basename "$d")
        # Find matching model file
        for ext in pt pkl; do
            p="outputs/models/${tag}.${ext}"
            [[ -f "$p" ]] && MODELS+=("$p")
        done
    done
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "ERROR: no model paths given. Use --all or pass at least one model path." >&2
    echo "Run with --help for usage." >&2
    exit 2
fi

# ─── Archive each model ────────────────────────────────────────────────────
for MODEL in "${MODELS[@]}"; do
    if [[ ! -f "$MODEL" ]]; then
        echo "⚠ skipping $MODEL (not found)"; continue
    fi
    BASE=$(basename "$MODEL")
    TAG="${BASE%.*}"

    SRC_REPORTS="outputs/reports/$TAG"
    SRC_SUBMISSIONS="outputs/submissions/$TAG"
    DEST="results/$TAG"

    if [[ ! -d "$SRC_REPORTS" ]] || [[ ! -d "$SRC_SUBMISSIONS" ]]; then
        echo "⚠ skipping $TAG — no reports or submissions found at $SRC_REPORTS / $SRC_SUBMISSIONS"
        echo "    (run ./scripts/evaluate_all.sh on this model first)"
        continue
    fi

    echo "[$(date '+%H:%M:%S')] archiving $TAG → $DEST/"
    mkdir -p "$DEST/metrics" "$DEST/submissions"

    cp "$SRC_REPORTS"/task*.json "$DEST/metrics/"
    cp "$SRC_SUBMISSIONS"/task*.csv "$DEST/submissions/" 2>/dev/null || true
    cp "$SRC_SUBMISSIONS"/task*.meta.json "$DEST/submissions/" 2>/dev/null || true

    if $INCLUDE_MODEL; then
        # If a .best.pt sibling exists, archive THAT one — it carries the
        # lowest-val-loss weights, which is the model we want to ship.
        if [[ "$MODEL" == *.pt && "$MODEL" != *.best.pt ]]; then
            BEST_SIBLING="${MODEL%.pt}.best.pt"
            if [[ -f "$BEST_SIBLING" ]]; then
                echo "    using best-val-loss sibling: $(basename "$BEST_SIBLING")"
                MODEL="$BEST_SIBLING"
            fi
        fi
        MODEL_EXT="${MODEL##*.}"
        if [[ "$MODEL_EXT" == "pt" ]]; then
            # Strip optimizer/scheduler/resume state — the archived file is the
            # final inference artifact, not a mid-training resume point. Roughly
            # halves the file size (drops the ~50 MB of Adam moments).
            "${PYTHON[@]}" - "$MODEL" "$DEST/model.pt" <<'PY'
import sys
from pathlib import Path
import torch

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
ckpt = torch.load(src, map_location="cpu", weights_only=False)

# Keep only what TransformerPredictor.load() reads (inference + identification).
keep = {"config", "state_dict", "tokenizer_data", "embedder_data",
        "unigram", "arch_kwargs", "wandb_run_id"}
slim = {k: v for k, v in ckpt.items() if k in keep}
slim["training_complete"] = True   # mark as finished, no resume state

torch.save(slim, dst)
src_mb = src.stat().st_size / 1e6
dst_mb = dst.stat().st_size / 1e6
saved = src_mb - dst_mb
pct = (saved / src_mb * 100) if src_mb > 0 else 0
print(f"    model stripped: {src_mb:.1f} MB → {dst_mb:.1f} MB  (saved {saved:.1f} MB / {pct:.0f}%)")
PY
        else
            # .pkl files (n-gram) — no optimizer state to strip, copy as-is.
            cp "$MODEL" "$DEST/model.$MODEL_EXT"
            ls -lh "$DEST/model.$MODEL_EXT" | awk '{print "    model copied:", $9, "("$5")"}'
        fi
    fi

    # ─── Generate README.md with metrics table + config ───────────────────
    "${PYTHON[@]}" - "$TAG" "$DEST" "$INCLUDE_MODEL" <<'PY'
import json, sys
from pathlib import Path
from datetime import datetime, timezone

tag = sys.argv[1]
dest = Path(sys.argv[2])
include_model = sys.argv[3].lower() == "true"

metrics_dir = dest / "metrics"
sub_dir = dest / "submissions"

# Read meta from any task — they all carry the same model_info.
meta_path = next(sub_dir.glob("task*.meta.json"), None)
meta = json.loads(meta_path.read_text()) if meta_path else {}
model_info = meta.get("model", {})
wandb_run_id = model_info.get("wandb_run_id")
wandb_link = (f"https://wandb.ai/fedja-bogataj-org/infineon-track1/runs/{wandb_run_id}"
              if wandb_run_id else None)

# Headline metrics per task (↑ = higher better, ↓ = lower better).
headline = {
    "task1.json": ("Next-step prediction", [
        ("top_1_accuracy", "↑"),
        ("top_3_accuracy", "↑"),
        ("top_5_accuracy", "↑"),
        ("mrr",            "↑"),
    ]),
    "task2.json": ("Sequence completion", [
        ("exact_match_rate",          "↑"),
        ("normalized_edit_distance",  "↓"),
        ("token_accuracy",            "↑"),
        ("block_accuracy",            "↑"),
    ]),
    "task3.json": ("Anomaly detection", [
        ("binary_accuracy",            "↑"),
        ("f1",                         "↑"),
        ("roc_auc",                    "↑"),
        ("rule_attribution_accuracy",  "↑"),
    ]),
}

lines = []
lines.append(f"# {tag}")
lines.append("")
lines.append(f"_Archived {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_")
lines.append("")

# Config block
lines.append("## Configuration")
lines.append("")
lines.append("| Setting | Value |")
lines.append("|---|---|")
lines.append(f"| Model type | `{model_info.get('type', '?')}` |")
if "order" in model_info:
    lines.append(f"| N-gram order | {model_info['order']} |")
if model_info.get("embeddings"):
    lines.append(f"| Embeddings | `{model_info['embeddings']}` |")
lines.append(f"| Seed | {meta.get('seed', '?')} |")
lines.append(f"| Eval split hash | `{meta.get('eval_split_hash', '?')}` |")
lines.append(f"| Submission timestamp | {meta.get('timestamp_utc', '?')} |")
if include_model:
    model_files = [p for p in dest.iterdir() if p.name.startswith("model.")]
    if model_files:
        size_mb = model_files[0].stat().st_size / 1e6
        lines.append(f"| Checkpoint | `{model_files[0].name}` ({size_mb:.1f} MB) |")
if wandb_link:
    lines.append(f"| W&B run | [{wandb_run_id}]({wandb_link}) |")
lines.append("")

# Metrics tables — one per task
for fn, (task_name, metrics) in headline.items():
    p = metrics_dir / fn
    if not p.exists():
        continue
    data = json.loads(p.read_text())
    overall = data.get("overall", {})
    n = overall.get("n", "?")
    lines.append(f"## {task_name}  (n={n})")
    lines.append("")
    lines.append("| Metric | Value | Better |")
    lines.append("|---|---|---|")
    for metric, direction in metrics:
        v = overall.get(metric)
        if isinstance(v, (int, float)):
            lines.append(f"| `{metric}` | **{v:.4f}** | {direction} |")
        else:
            lines.append(f"| `{metric}` | — | {direction} |")
    # Confusion matrix for anomaly
    if fn == "task3.json":
        cm = overall.get("confusion", {})
        if cm:
            lines.append("")
            lines.append("Confusion matrix (positive = invalid):")
            lines.append("")
            lines.append(f"|        | predicted invalid | predicted valid |")
            lines.append(f"|---|---|---|")
            lines.append(f"| **actually invalid** | TP = {cm.get('tp','?')} | FN = {cm.get('fn','?')} |")
            lines.append(f"| **actually valid**   | FP = {cm.get('fp','?')} | TN = {cm.get('tn','?')} |")
    lines.append("")

# Per-family breakdown (currently empty in our reports but ready for when it lands)
any_family = False
for fn in headline:
    p = metrics_dir / fn
    if p.exists():
        d = json.loads(p.read_text())
        if d.get("per_family"):
            any_family = True
            break
if any_family:
    lines.append("## Per-family breakdown")
    lines.append("")
    for fn, (task_name, _) in headline.items():
        p = metrics_dir / fn
        if not p.exists(): continue
        d = json.loads(p.read_text())
        fam = d.get("per_family", {})
        if not fam: continue
        lines.append(f"### {task_name}")
        lines.append("")
        fams = sorted(fam)
        cols = list(headline[fn][1])
        header = "| metric | " + " | ".join(fams) + " |"
        sep = "|---|" + "|".join(["---"] * len(fams)) + "|"
        lines.append(header); lines.append(sep)
        for metric, direction in cols:
            row = [f"`{metric}` ({direction})"]
            for f in fams:
                v = fam.get(f, {}).get(metric)
                row.append(f"{v:.4f}" if isinstance(v, (int, float)) else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

lines.append("---")
lines.append("")
lines.append(f"To reproduce: see commit history and `scripts/train_transformer.sbatch`.")
lines.append(f"Submission CSVs and metric JSONs in this folder are deterministic from "
             f"the model checkpoint + the canonical eval set at `outputs/eval_canonical/`.")
lines.append("")

(dest / "README.md").write_text("\n".join(lines))
print(f"    ✓ README.md generated ({sum(1 for l in lines):d} lines)")
PY
done

# ─── Refresh top-level results/README.md catalogue ─────────────────────────
"${PYTHON[@]}" - <<'PY'
import json
from pathlib import Path
from datetime import datetime, timezone

results_dir = Path("results")
runs = sorted([p for p in results_dir.iterdir() if p.is_dir() and (p / "metrics/task1.json").exists()])

if not runs:
    raise SystemExit(0)

lines = [
    "# Archived experiment results",
    "",
    f"_Last updated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
    "",
    "Each subdirectory is a single archived experiment. Follow its `README.md`",
    "for the full metrics table; this top-level table shows headline numbers across runs.",
    "",
    "## Cross-experiment comparison — headline metrics",
    "",
]

# Build a compact comparison table
headline_keys = [
    ("task1.json", "top_1_accuracy", "Top-1 (↑)"),
    ("task1.json", "top_5_accuracy", "Top-5 (↑)"),
    ("task1.json", "mrr",            "MRR (↑)"),
    ("task2.json", "normalized_edit_distance", "NED (↓)"),
    ("task2.json", "block_accuracy", "Block acc (↑)"),
    ("task3.json", "f1",             "F1 (↑)"),
    ("task3.json", "roc_auc",        "ROC-AUC (↑)"),
]

header = "| run | " + " | ".join(label for _,_,label in headline_keys) + " |"
sep = "|---|" + "|".join(["---"] * len(headline_keys)) + "|"
lines += [header, sep]
for run in runs:
    row = [f"[`{run.name}`]({run.name}/)"]
    for fn, metric, _ in headline_keys:
        p = run / "metrics" / fn
        if not p.exists():
            row.append("—"); continue
        d = json.loads(p.read_text())
        v = d.get("overall", {}).get(metric)
        row.append(f"{v:.4f}" if isinstance(v, (int, float)) else "—")
    lines.append("| " + " | ".join(row) + " |")

lines += ["", "_Best per column is whatever maximises the arrow direction. NED (↓) is lower-is-better; the rest are higher-is-better._"]

(results_dir / "README.md").write_text("\n".join(lines))
print(f"✓ results/README.md updated ({len(runs)} runs catalogued)")
PY

echo
echo "Done. Stage with:  git add results/"
