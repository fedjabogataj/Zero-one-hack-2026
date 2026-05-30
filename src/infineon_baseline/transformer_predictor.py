"""NGram-compatible wrapper around a trained TransformerLM.

Exposes the same attributes and methods as `NGram` so
`predictor.run_task1/2/3` work without modification:
  - model.order       (int)
  - model.unigram     (dict[str, Counter])
  - model.top_k(family, prefix, k) -> list[int]
  - model.log_prob(family, ids) -> float

Also exposes batched fast paths used by predictor.py when batch_size > 1:
  - model.top_k_batch(family_list, prefixes, k) -> list[list[int]]
  - model.log_prob_batch(family_list, ids_list) -> list[float]
  - model.generate_batch(family_list, prefixes, max_steps, stop_id) -> list[list[int]]

OOD step aliasing: when a step name is not in the tokenizer vocabulary,
we use STStepEmbedder.nearest() to map it to the closest known step,
mirroring the SoftNGram.alias_step_name() behaviour.
"""
from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


# Inference dtypes that the model can be cast to. bf16 on Ampere+ (A100, H100)
# gives ~2x throughput with no accuracy loss for this model size. fp16 also
# works on Volta/Turing/Ampere but is less numerically robust. fp32 is the
# safe default everywhere (and the only sensible option on CPU).
_DTYPE_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


class TransformerPredictor:
    """Wraps TransformerLM with the NGram interface for use in run_task*."""

    order: int          # set to max_seq_len so run_task1 passes the full prefix
    unigram: dict[str, Counter]

    def __init__(
        self,
        model,          # TransformerLM (already on device, eval mode)
        tokenizer,      # Tokenizer
        embedder,       # STStepEmbedder
        unigram: dict[str, Counter],
        device: torch.device,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._embedder = embedder
        self.unigram = unigram
        self._device = device
        # order = max_seq_len so that run_task1 slices prefix[-(order-1):]
        # which will just be the full prefix for any real sequence.
        self.order = model.config.max_seq_len

        # Precompute per-family vocabulary masks so batched top_k can mask
        # out-of-family steps with a single masked_fill_ instead of a
        # per-row Python loop. A True entry == "mask this slot to -inf".
        pad_id = model.config.pad_id
        vocab_plus_pad = pad_id + 1
        self._family_mask: dict[str, torch.Tensor] = {}
        for fam, ctr in self.unigram.items():
            m = torch.ones(vocab_plus_pad, dtype=torch.bool, device=device)
            for sid in ctr:
                if 0 <= sid < pad_id:
                    m[sid] = False
            m[pad_id] = True  # PAD slot is always masked
            self._family_mask[fam] = m
        # Fallback for unknown families: only PAD is masked, all real steps allowed.
        fallback = torch.zeros(vocab_plus_pad, dtype=torch.bool, device=device)
        fallback[pad_id] = True
        self._family_mask_fallback = fallback

    @classmethod
    def load(
        cls,
        path: Path,
        device: str = "auto",
        precision: str = "auto",
    ) -> "TransformerPredictor":
        """Load a self-contained checkpoint produced by `train.py`.

        precision: "auto" → bf16 on CUDA, fp32 elsewhere; or explicit
        "fp32" / "bf16" / "fp16".
        """
        from infineon_baseline.tokenizer import Tokenizer
        from infineon_baseline.embeddings_st import STStepEmbedder
        from infineon_baseline.transformer_model import TransformerLM, TransformerConfig

        if device == "auto":
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            _device = torch.device(device)

        if precision == "auto":
            precision = "bf16" if _device.type == "cuda" else "fp32"
        dtype = _DTYPE_MAP[precision]

        ckpt = torch.load(path, map_location=_device, weights_only=False)

        # Reconstruct tokenizer from embedded data.
        tok_data = ckpt["tokenizer_data"]
        id_to_step = list(tok_data["id_to_step"])
        tokenizer = Tokenizer(
            step_to_id={s: i for i, s in enumerate(id_to_step)},
            id_to_step=id_to_step,
            family_to_id=dict(tok_data["family_to_id"]),
        )

        # Reconstruct embedder from embedded data.
        emb_data = pickle.loads(ckpt["embedder_data"])
        embedder = STStepEmbedder(
            vectors=emb_data["vectors"],
            step_to_idx=emb_data["step_to_idx"],
            idx_to_step=emb_data["idx_to_step"],
            _row_norms=emb_data["row_norms"],
        )

        # Reconstruct model.
        cfg_dict = ckpt["config"]
        config = TransformerConfig(**cfg_dict)
        model = TransformerLM(config)
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(_device)
        if dtype is not torch.float32:
            model = model.to(dtype)
        model.eval()

        unigram = {fam: Counter(ctr) for fam, ctr in ckpt["unigram"].items()}
        return cls(model=model, tokenizer=tokenizer, embedder=embedder,
                   unigram=unigram, device=_device)

    # ─── OOD aliasing ─────────────────────────────────────────────────── #

    def _alias_step(self, step: str) -> int:
        """Map a step name to a token id, using nearest-neighbor for OOD names."""
        sid = self._tokenizer.step_to_id.get(step)
        if sid is not None:
            return sid
        nearest = self._embedder.nearest(step, k=1, exclude_self=False)
        if nearest:
            return self._tokenizer.step_to_id.get(nearest[0][0], 0)
        return 0   # last-resort fallback: token 0

    def _encode_prefix(self, family: str, prefix: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (family_id_tensor, step_ids_tensor) for a single inference call."""
        fid = self._tokenizer.family_to_id.get(family, 0)
        family_t = torch.tensor([fid], dtype=torch.long, device=self._device)
        if len(prefix) == 0:
            # Empty prefix: feed a single PAD token so forward() gets valid input.
            pad_id = self._model.config.pad_id
            ids_t = torch.tensor([[pad_id]], dtype=torch.long, device=self._device)
        else:
            max_len = self._model.config.max_seq_len - 1
            ids = list(prefix[-max_len:])
            ids_t = torch.tensor([ids], dtype=torch.long, device=self._device)
        return family_t, ids_t

    # ─── NGram-compatible (single-example) interface ─────────────────── #

    @torch.no_grad()
    def top_k(self, family: str, prefix: tuple[int, ...], k: int = 5) -> list[int]:
        """Return the k most-likely next-step ids."""
        return self.top_k_batch([family], [prefix], k=k)[0]

    @torch.no_grad()
    def log_prob(self, family: str, ids: list[int]) -> float:
        """Sum of log P(id_i | prefix_i) under the model (teacher-forcing)."""
        return self.log_prob_batch([family], [ids])[0]

    # ─── Batched fast path ───────────────────────────────────────────── #

    def _pack_batch(
        self,
        sequences: Sequence[Sequence[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Right-pad a list of int sequences to a (B, T) tensor.

        Returns (step_ids, key_padding_mask, last_indices) where
          - step_ids       (B, T) long
          - key_padding_mask (B, T) bool — True = PAD position (ignored by attention)
          - last_indices   (B,)   long — index of the last real token in each row
                                          (used to gather "next-step" logits).

        Empty sequences are encoded as a single PAD (unmasked) so the model
        still gets at least one position to attend to; their last_index is 0.
        """
        B = len(sequences)
        pad_id = self._model.config.pad_id
        max_seq_len = self._model.config.max_seq_len

        # Truncate each prefix to the last (max_seq_len - 1) tokens for top_k,
        # or max_seq_len for log_prob — caller decides the cap by pre-clipping.
        # Here we only enforce the hard model cap so attention never overflows.
        lens = [max(1, min(len(s), max_seq_len)) for s in sequences]
        T = max(lens)

        step_ids = torch.full((B, T), pad_id, dtype=torch.long, device=self._device)
        pad_mask = torch.ones((B, T), dtype=torch.bool, device=self._device)
        last_idx = torch.empty(B, dtype=torch.long, device=self._device)

        for i, s in enumerate(sequences):
            L = lens[i]
            if len(s) == 0:
                # Empty: leave PAD in slot 0, unmask it so attention is well-defined.
                pad_mask[i, 0] = False
                last_idx[i] = 0
            else:
                clipped = list(s[-L:])
                step_ids[i, :L] = torch.tensor(clipped, dtype=torch.long, device=self._device)
                pad_mask[i, :L] = False
                last_idx[i] = L - 1
        return step_ids, pad_mask, last_idx

    def _family_id_tensor(self, family_list: Sequence[str]) -> torch.Tensor:
        ids = [self._tokenizer.family_to_id.get(f, 0) for f in family_list]
        return torch.tensor(ids, dtype=torch.long, device=self._device)

    def _stack_family_masks(self, family_list: Sequence[str]) -> torch.Tensor:
        """(B, V+1) bool mask: True = forbidden slot (PAD or out-of-family)."""
        rows = [self._family_mask.get(f, self._family_mask_fallback) for f in family_list]
        return torch.stack(rows, dim=0)

    @torch.no_grad()
    def top_k_batch(
        self,
        family_list: Sequence[str],
        prefixes: Sequence[Sequence[int]],
        k: int = 5,
    ) -> list[list[int]]:
        """Batched version of top_k.

        For each example, returns the k highest-logit next-step ids after
        masking PAD and steps unseen in that family.
        """
        assert len(family_list) == len(prefixes)
        if not family_list:
            return []
        self._model.eval()

        # Reserve one position for the "next token" we want to predict by
        # capping prefixes to max_seq_len - 1.
        max_seq_len = self._model.config.max_seq_len
        prefixes = [list(p[-(max_seq_len - 1):]) for p in prefixes]

        step_ids, pad_mask, last_idx = self._pack_batch(prefixes)
        family_ids = self._family_id_tensor(family_list)

        logits = self._model(family_ids, step_ids, key_padding_mask=pad_mask)  # (B, T, V+1)
        # Gather logits at each example's last position.
        batch_idx = torch.arange(logits.shape[0], device=self._device)
        last_logits = logits[batch_idx, last_idx, :].float()  # (B, V+1), upcast for stable topk

        # Mask PAD + out-of-family steps.
        masks = self._stack_family_masks(family_list)
        last_logits = last_logits.masked_fill(masks, float("-inf"))

        pad_id = self._model.config.pad_id
        top = torch.topk(last_logits, k=min(k, pad_id), dim=-1).indices  # (B, k)
        return top.tolist()

    @torch.no_grad()
    def log_prob_batch(
        self,
        family_list: Sequence[str],
        ids_list: Sequence[Sequence[int]],
    ) -> list[float]:
        """Batched version of log_prob (teacher-forced)."""
        assert len(family_list) == len(ids_list)
        if not family_list:
            return []
        self._model.eval()

        pad_id = self._model.config.pad_id
        max_seq_len = self._model.config.max_seq_len
        clipped = [list(ids)[:max_seq_len] for ids in ids_list]
        # Input is [PAD] + ids[:-1]; target is ids. Position t predicts target[t].
        inputs = [[pad_id] + ids[:-1] if ids else [pad_id] for ids in clipped]

        step_ids, pad_mask, _ = self._pack_batch(inputs)
        family_ids = self._family_id_tensor(family_list)

        logits = self._model(family_ids, step_ids, key_padding_mask=pad_mask)  # (B, T, V+1)
        # Upcast to fp32 for softmax — log-prob accuracy matters more here than
        # the bf16 throughput savings on a single softmax.
        log_probs = F.log_softmax(logits.float(), dim=-1)

        out: list[float] = []
        for i, tgt in enumerate(clipped):
            if not tgt:
                out.append(0.0)
                continue
            L = len(tgt)
            tgt_t = torch.tensor(tgt, dtype=torch.long, device=self._device)
            total = log_probs[i, :L, :].gather(1, tgt_t.unsqueeze(1)).sum().item()
            out.append(float(total))
        return out

    @torch.no_grad()
    def generate_batch(
        self,
        family_list: Sequence[str],
        prefixes: Sequence[Sequence[int]],
        max_steps_list: Sequence[int],
        stop_id: int | None,
    ) -> list[list[int]]:
        """Greedy autoregressive generation for B examples in parallel.

        Each example finishes when it (a) emits stop_id or (b) hits its
        per-example max_steps. Finished examples are removed from the active
        batch so we stop wasting compute on them.

        Returns per-example list of generated ids (NOT including the prefix).
        """
        assert len(family_list) == len(prefixes) == len(max_steps_list)
        B = len(family_list)
        outputs: list[list[int]] = [[] for _ in range(B)]
        if B == 0:
            return outputs

        # Mutable working state per example.
        prefix_state = [list(p) for p in prefixes]
        max_steps = list(max_steps_list)
        active = list(range(B))   # indices still generating

        while active:
            sub_fams = [family_list[i] for i in active]
            sub_prefs = [prefix_state[i] for i in active]
            next_ids_batch = self.top_k_batch(sub_fams, sub_prefs, k=1)

            new_active: list[int] = []
            for slot, i in enumerate(active):
                nid = next_ids_batch[slot][0]
                outputs[i].append(nid)
                prefix_state[i].append(nid)
                if stop_id is not None and nid == stop_id:
                    continue
                if len(outputs[i]) >= max_steps[i]:
                    continue
                new_active.append(i)
            active = new_active
        return outputs
