"""Training loop for the TransformerLM next-step model.

Reads train_split.csv (produced by `build-eval`), tokenises sequences, and
trains a TransformerLM with causal LM objective. Saves a self-contained
checkpoint (.pt) that TransformerPredictor can load without separate files.

Usage (via CLI):
    infineon-baseline train \\
        --train outputs/eval/train_split.csv \\
        --embeddings outputs/models/st_step_embeddings.pkl \\
        --out outputs/models/transformer.pt \\
        --epochs 20 --batch-size 32 --lr 3e-4 --device auto --seed 42
"""
from __future__ import annotations

import argparse
import csv
import math
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ─── Dataset ──────────────────────────────────────────────────────────────── #

class SequenceDataset(Dataset):
    """One sample = one manufacturing sequence as (family_id, ids_list)."""

    def __init__(
        self,
        samples: list[tuple[int, list[int]]],   # (family_id, id_list)
        max_seq_len: int,
        pad_id: int,
    ) -> None:
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        family_id, ids = self.samples[idx]
        # Truncate to max_seq_len (keep first tokens; sequence starts are most informative).
        ids = ids[:self.max_seq_len]
        T = len(ids)
        # Input: ids[:-1], target: ids[1:] (next-token prediction).
        inp = ids[:-1] if T > 1 else ids
        tgt = ids[1:] if T > 1 else ids
        inp_len = len(inp)
        # Pad to max_seq_len - 1.
        max_T = self.max_seq_len - 1
        inp_t = torch.full((max_T,), self.pad_id, dtype=torch.long)
        tgt_t = torch.full((max_T,), -100, dtype=torch.long)  # -100 = ignore in CE loss
        inp_t[:inp_len] = torch.tensor(inp, dtype=torch.long)
        tgt_t[:inp_len] = torch.tensor(tgt, dtype=torch.long)
        pad_mask = torch.zeros(max_T, dtype=torch.bool)
        pad_mask[inp_len:] = True  # True = padding position (ignored in attention)
        return {
            "family_id": torch.tensor(family_id, dtype=torch.long),
            "step_ids": inp_t,
            "targets": tgt_t,
            "pad_mask": pad_mask,
        }


def _load_train_split(csv_path: Path) -> dict[str, dict[str, list[str]]]:
    """Read train_split.csv -> {family: {seq_id: [steps]}}."""
    data: dict[str, dict[str, list[str]]] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            data.setdefault(row["FAMILY"], {}).setdefault(row["SEQUENCE_ID"], []).append(row["STEP"])
    return data


def _build_samples(
    corpus: dict[str, dict[str, list[str]]],
    tokenizer,
    max_seq_len: int,
    pad_id: int,
) -> tuple[list[tuple[int, list[int]]], dict[str, Counter]]:
    """Tokenise all sequences; also compute unigram counts (for TransformerPredictor)."""
    samples: list[tuple[int, list[int]]] = []
    unigram: dict[str, Counter] = {}
    for family, seqs in corpus.items():
        family_id = tokenizer.family_to_id[family]
        uc: Counter = Counter()
        for steps in seqs.values():
            # Skip steps not in vocabulary (shouldn't happen for train data).
            ids = [tokenizer.step_to_id[s] for s in steps if s in tokenizer.step_to_id]
            if len(ids) < 2:
                continue
            uc.update(ids)
            samples.append((family_id, ids))
        unigram[family] = uc
    return samples, unigram


# ─── Scheduler ────────────────────────────────────────────────────────────── #

def _make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup for `warmup_steps` then cosine decay to 0."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Training entrypoint ──────────────────────────────────────────────────── #

def train(args: argparse.Namespace) -> int:
    """Main training loop. Called by the CLI's `train` subcommand."""
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Device ──────────────────────────────────────────────────────────── #
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load data ───────────────────────────────────────────────────────── #
    from infineon_baseline.tokenizer import Tokenizer
    from infineon_baseline.embeddings_st import STStepEmbedder
    from infineon_baseline.transformer_model import TransformerLM

    corpus = _load_train_split(Path(args.train))
    # Build tokenizer from the corpus so it covers exactly the training vocabulary.
    tokenizer = Tokenizer.fit(corpus)
    embedder = STStepEmbedder.load(Path(args.embeddings))

    max_seq_len = getattr(args, "max_seq_len", 256)
    vocab_size = len(tokenizer.id_to_step)
    pad_id = vocab_size  # one slot beyond the actual vocab

    all_samples, unigram = _build_samples(corpus, tokenizer, max_seq_len, pad_id)

    # Val split: last 10% of samples (deterministic, no shuffle needed).
    n_val = max(1, len(all_samples) // 10)
    train_samples = all_samples[:-n_val]
    val_samples = all_samples[-n_val:]
    print(f"Train sequences: {len(train_samples)}  Val: {len(val_samples)}")

    train_ds = SequenceDataset(train_samples, max_seq_len, pad_id)
    val_ds = SequenceDataset(val_samples, max_seq_len, pad_id)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=g, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model ───────────────────────────────────────────────────────────── #
    arch_kwargs: dict = {}
    for key in ("d_model", "n_heads", "n_layers", "ff_dim", "dropout"):
        val = getattr(args, key, None)
        if val is not None:
            arch_kwargs[key] = val
    arch_kwargs["max_seq_len"] = max_seq_len

    model = TransformerLM.from_tokenizer_and_embedder(tokenizer, embedder, **arch_kwargs)
    model = model.to(device)
    print(f"Params: {model.param_count():,}")

    # ── Optimizer & scheduler ───────────────────────────────────────────── #
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=0.01, betas=(0.9, 0.95),
    )
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(200, total_steps // 10)
    scheduler = _make_scheduler(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # ── Training loop ───────────────────────────────────────────────────── #
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        n_tokens = 0

        for batch in train_loader:
            family_ids = batch["family_id"].to(device)
            step_ids = batch["step_ids"].to(device)
            targets = batch["targets"].to(device)
            pad_mask = batch["pad_mask"].to(device)

            logits = model(family_ids, step_ids, key_padding_mask=pad_mask)
            # logits: (B, T, V+1); targets: (B, T) with -100 for PAD positions
            B, T, Vp1 = logits.shape
            loss = criterion(logits.view(B * T, Vp1), targets.view(B * T))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            batch_tokens = int((targets != -100).sum().item())
            n_tokens += batch_tokens
            epoch_loss += loss.item()

            if global_step == 1:
                elapsed = time.time() - t0
                print(f"  First step: {elapsed:.2f}s, "
                      f"tokens/s approx {batch_tokens / max(elapsed, 1e-6):.0f}")

            if global_step % 50 == 0:
                print(f"  step {global_step:5d} | loss {loss.item():.4f} "
                      f"| lr {scheduler.get_last_lr()[0]:.2e}")

            if global_step % 500 == 0:
                val_loss = _eval(model, val_loader, criterion, device)
                print(f"  [val step {global_step}] loss {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch}/{args.epochs} | avg_loss {avg_loss:.4f} "
              f"| {n_tokens / max(elapsed, 1e-6):.0f} tok/s | {elapsed:.1f}s")

    # Final val pass.
    val_loss = _eval(model, val_loader, criterion, device)
    print(f"Final val loss: {val_loss:.4f}")

    # ── Save checkpoint ─────────────────────────────────────────────────── #
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialise tokenizer and embedder inline — checkpoint is fully self-contained.
    tokenizer_data = {
        "id_to_step": tokenizer.id_to_step,
        "family_to_id": tokenizer.family_to_id,
    }
    embedder_data = pickle.dumps({
        "vectors": embedder.vectors,
        "step_to_idx": embedder.step_to_idx,
        "idx_to_step": embedder.idx_to_step,
        "row_norms": embedder._row_norms,
    })

    checkpoint = {
        "config": vars(model.config),
        "state_dict": model.state_dict(),
        "tokenizer_data": tokenizer_data,
        "embedder_data": embedder_data,
        "unigram": {fam: dict(ctr) for fam, ctr in unigram.items()},
        "arch_kwargs": arch_kwargs,
    }
    torch.save(checkpoint, out_path)
    print(f"✓ checkpoint saved → {out_path}")
    return 0


def _eval(model, loader, criterion, device) -> float:
    """Run one pass over `loader` and return mean cross-entropy loss."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            family_ids = batch["family_id"].to(device)
            step_ids = batch["step_ids"].to(device)
            targets = batch["targets"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            logits = model(family_ids, step_ids, key_padding_mask=pad_mask)
            B, T, Vp1 = logits.shape
            loss = criterion(logits.view(B * T, Vp1), targets.view(B * T))
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)
