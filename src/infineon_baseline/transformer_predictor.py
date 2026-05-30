"""NGram-compatible wrapper around a trained TransformerLM.

Two operating modes, selected from the checkpoint's `tokenizer_kind`:

* **flat** — one token per step string (vocab ~120). Exposes the NGram-style
  interface so predictor.run_task1/2/3 can reuse the flat dispatch path:
  - model.order, model.unigram
  - model.top_k(family, prefix, k) -> list[int]
  - model.log_prob(family, ids)   -> float

* **subword** — word-level subwords with <sep>/<eos> boundaries. Exposes
  step-level methods that hide the subword decoding from predictor.run_*:
  - model.top_k_steps(family, partial_steps, k) -> list[str]   (beam search)
  - model.complete_steps(family, partial_steps, max_steps) -> list[str]
  - model.log_prob_steps(family, steps) -> float

Both modes also expose batched fast paths used by predictor.py when
batch_size > 1 — typically 10-50x speedup on A100 over the per-example loop:

  flat:    top_k_batch       / log_prob_batch    / generate_batch
  subword: top_k_steps_batch / log_prob_steps_batch / complete_steps_batch

OOD step aliasing (flat only): unknown step names map to the nearest known
step via STStepEmbedder.nearest(), mirroring SoftNGram.alias_step_name().
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
        tokenizer,      # Tokenizer or SubwordTokenizer
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
        # Detect subword mode from tokenizer duck-type.
        self._is_subword: bool = hasattr(tokenizer, "sep_id")

        # Precompute per-family vocabulary masks for the FLAT-mode batched
        # top_k path. In subword mode the mask is over subword tokens, which
        # would forbid emitting <sep>/<eos>/unseen-subwords needed for OOD
        # decoding — so subword methods skip masking entirely.
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

        Reads `tokenizer_kind` from the checkpoint (defaults to "flat" for
        backward compatibility with old checkpoints) and reconstructs the
        appropriate tokenizer class.

        precision: "auto" -> bf16 on CUDA, fp32 elsewhere; or "fp32" / "bf16" / "fp16".
        """
        from infineon_baseline.tokenizer import Tokenizer
        from infineon_baseline.subword_tokenizer import SubwordTokenizer
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

        tokenizer_kind = ckpt.get("tokenizer_kind", "flat")
        tok_data = ckpt["tokenizer_data"]

        if tokenizer_kind == "subword":
            id_to_token = list(tok_data["id_to_token"])
            tokenizer = SubwordTokenizer(
                token_to_id={t: i for i, t in enumerate(id_to_token)},
                id_to_token=id_to_token,
                family_to_id=dict(tok_data["family_to_id"]),
            )
        else:
            # Flat tokenizer (default / backward compat).
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
        instance = cls(model=model, tokenizer=tokenizer, embedder=embedder,
                       unigram=unigram, device=_device)
        instance._is_subword = (tokenizer_kind == "subword")
        return instance

    # ─── OOD aliasing (flat mode only) ────────────────────────────────── #

    def _alias_step(self, step: str) -> int:
        """Map a step name to a token id, using nearest-neighbor for OOD names.

        Only used in flat tokenizer mode. In subword mode, OOD handling is
        implicit: unknown subwords map to <unk>.
        """
        sid = self._tokenizer.step_to_id.get(step)
        if sid is not None:
            return sid
        nearest = self._embedder.nearest(step, k=1, exclude_self=False)
        if nearest:
            return self._tokenizer.step_to_id.get(nearest[0][0], 0)
        return 0   # last-resort fallback: token 0

    # ─── Batching primitives (used by both flat & subword fast paths) ── #

    def _family_id_tensor(self, family_list: Sequence[str]) -> torch.Tensor:
        ids = [self._tokenizer.family_to_id.get(f, 0) for f in family_list]
        return torch.tensor(ids, dtype=torch.long, device=self._device)

    def _stack_family_masks(self, family_list: Sequence[str]) -> torch.Tensor:
        """(B, V+1) bool mask: True = forbidden slot (PAD or out-of-family)."""
        rows = [self._family_mask.get(f, self._family_mask_fallback) for f in family_list]
        return torch.stack(rows, dim=0)

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

        Empty sequences are encoded as a single PAD (unmasked) so attention
        always has at least one position to attend to.
        """
        B = len(sequences)
        pad_id = self._model.config.pad_id
        max_seq_len = self._model.config.max_seq_len

        # Cap each row at the model's max sequence length. Caller should
        # pre-clip prefixes if it wants headroom for a "next" token.
        lens = [max(1, min(len(s), max_seq_len)) for s in sequences]
        T = max(lens)

        step_ids = torch.full((B, T), pad_id, dtype=torch.long, device=self._device)
        pad_mask = torch.ones((B, T), dtype=torch.bool, device=self._device)
        last_idx = torch.empty(B, dtype=torch.long, device=self._device)

        for i, s in enumerate(sequences):
            L = lens[i]
            if len(s) == 0:
                pad_mask[i, 0] = False
                last_idx[i] = 0
            else:
                clipped = list(s[-L:])
                step_ids[i, :L] = torch.tensor(clipped, dtype=torch.long, device=self._device)
                pad_mask[i, :L] = False
                last_idx[i] = L - 1
        return step_ids, pad_mask, last_idx

    @torch.no_grad()
    def _last_logits_batch(
        self,
        family_list: Sequence[str],
        prefixes: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Raw (B, V+1) logits at the last real position of each prefix.

        Returned as fp32 regardless of model dtype, so downstream
        log_softmax / topk are numerically stable.
        """
        assert len(family_list) == len(prefixes)
        if not family_list:
            return torch.empty(0, self._model.config.pad_id + 1, device=self._device)
        self._model.eval()
        max_seq_len = self._model.config.max_seq_len
        # Leave room for one more token to be predicted.
        prefixes = [list(p[-(max_seq_len - 1):]) for p in prefixes]
        step_ids, pad_mask, last_idx = self._pack_batch(prefixes)
        family_ids = self._family_id_tensor(family_list)
        logits = self._model(family_ids, step_ids, key_padding_mask=pad_mask)
        batch_idx = torch.arange(logits.shape[0], device=self._device)
        return logits[batch_idx, last_idx, :].float()

    # ─── Flat-mode: single-example interface (NGram-compatible) ──────── #

    @torch.no_grad()
    def top_k(self, family: str, prefix: tuple[int, ...], k: int = 5) -> list[int]:
        """Return the k most-likely next-step ids (flat mode)."""
        return self.top_k_batch([family], [prefix], k=k)[0]

    @torch.no_grad()
    def log_prob(self, family: str, ids: list[int]) -> float:
        """Sum of log P(id_i | prefix_i) under the model (flat mode, teacher-forced)."""
        return self.log_prob_batch([family], [ids])[0]

    # ─── Flat-mode: batched interface ────────────────────────────────── #

    @torch.no_grad()
    def top_k_batch(
        self,
        family_list: Sequence[str],
        prefixes: Sequence[Sequence[int]],
        k: int = 5,
    ) -> list[list[int]]:
        """Batched top-k next-step with PAD + per-family masking (flat mode)."""
        if not family_list:
            return []
        last_logits = self._last_logits_batch(family_list, prefixes)  # (B, V+1)
        masks = self._stack_family_masks(family_list)
        last_logits = last_logits.masked_fill(masks, float("-inf"))
        pad_id = self._model.config.pad_id
        top = torch.topk(last_logits, k=min(k, pad_id), dim=-1).indices
        return top.tolist()

    @torch.no_grad()
    def log_prob_batch(
        self,
        family_list: Sequence[str],
        ids_list: Sequence[Sequence[int]],
    ) -> list[float]:
        """Batched teacher-forced log-prob over id sequences."""
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
        logits = self._model(family_ids, step_ids, key_padding_mask=pad_mask)
        # Upcast for stable log-softmax — single op, cost negligible.
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
        """Greedy autoregressive generation for B examples in parallel (flat mode).

        Each example finishes on (a) stop_id emitted or (b) per-example max_steps.
        Finished examples drop out of the active batch so we stop wasting compute.
        Returns per-example list of generated ids (NOT including the prefix).
        """
        assert len(family_list) == len(prefixes) == len(max_steps_list)
        B = len(family_list)
        outputs: list[list[int]] = [[] for _ in range(B)]
        if B == 0:
            return outputs

        prefix_state = [list(p) for p in prefixes]
        max_steps = list(max_steps_list)
        active = list(range(B))

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

    # ─── Subword-mode methods (single-example, kept for compat) ──────── #

    @torch.no_grad()
    def top_k_steps(self, family: str, partial_steps: list[str], k: int = 5) -> list[str]:
        """Beam search to find the k most-likely next step strings (subword mode)."""
        return self.top_k_steps_batch([family], [partial_steps], k=k)[0]

    @torch.no_grad()
    def complete_steps(
        self, family: str, partial_steps: list[str], max_steps: int = 200,
    ) -> list[str]:
        """Greedy autoregressive completion from the subword prefix (subword mode)."""
        return self.complete_steps_batch([family], [partial_steps], [max_steps])[0]

    @torch.no_grad()
    def log_prob_steps(self, family: str, steps: list[str]) -> float:
        """Teacher-forcing log-probability of a full step sequence (subword mode)."""
        if not steps:
            return 0.0
        return self.log_prob_steps_batch([family], [steps])[0]

    # ─── Subword-mode batched fast path ──────────────────────────────── #

    @torch.no_grad()
    def top_k_steps_batch(
        self,
        family_list: Sequence[str],
        partial_steps_list: Sequence[Sequence[str]],
        k: int = 5,
    ) -> list[list[str]]:
        """Batched beam search: top-k next step strings per example (subword mode).

        Each iteration flattens ALL active beams across ALL examples into one
        forward pass, then redistributes the resulting logits back per example.
        Beam-search width = k. Hard cap of max_step_len subword tokens per step.
        """
        assert len(family_list) == len(partial_steps_list)
        B = len(family_list)
        if B == 0:
            return []

        tok = self._tokenizer
        if not hasattr(tok, "sep_id"):
            raise RuntimeError("top_k_steps_batch requires a subword tokenizer")

        beam_size = k
        max_step_len = 16

        # Per-example state.
        # active_beams[i] = list of (input_ids, gen_ids, log_prob) still generating
        # finished[i]     = list of (gen_ids, log_prob) that hit sep/eos
        active_beams: list[list[tuple[list[int], list[int], float]]] = []
        finished: list[list[tuple[list[int], float]]] = [[] for _ in range(B)]
        for fam, partial in zip(family_list, partial_steps_list):
            _fid, prefix_ids = tok.encode_prefix(fam, list(partial))
            active_beams.append([(list(prefix_ids), [], 0.0)])

        for _ in range(max_step_len):
            # Flatten active beams across all examples into one batch.
            batch_fams: list[str] = []
            batch_prefixes: list[list[int]] = []
            batch_origin: list[tuple[int, int]] = []   # (ex_idx, beam_idx_within_ex)
            for ex_idx in range(B):
                for beam_idx, (input_ids, _gen, _lp) in enumerate(active_beams[ex_idx]):
                    batch_fams.append(family_list[ex_idx])
                    batch_prefixes.append(input_ids)
                    batch_origin.append((ex_idx, beam_idx))

            if not batch_prefixes:
                break

            # One big forward pass for all active beams.
            last_logits = self._last_logits_batch(batch_fams, batch_prefixes)  # (N, V+1)
            log_probs = F.log_softmax(last_logits, dim=-1)
            topv, topi = log_probs.topk(beam_size + 1, dim=-1)
            topv_list = topv.tolist()
            topi_list = topi.tolist()

            # Build candidate beams per example.
            new_per_ex: list[list[tuple[list[int], list[int], float]]] = [[] for _ in range(B)]
            for row_idx, (ex_idx, beam_idx) in enumerate(batch_origin):
                input_ids, gen, lp = active_beams[ex_idx][beam_idx]
                for v, i in zip(topv_list[row_idx], topi_list[row_idx]):
                    if i in (tok.sep_id, tok.eos_id):
                        # Reaching a boundary finalises the beam.
                        finished[ex_idx].append((list(gen), lp + v))
                    else:
                        new_per_ex[ex_idx].append((input_ids + [i], gen + [i], lp + v))

            # Per example: keep top-beam_size candidates by log-prob.
            for ex_idx in range(B):
                new_per_ex[ex_idx].sort(key=lambda x: -x[2])
                active_beams[ex_idx] = new_per_ex[ex_idx][:beam_size]

            # Early stop: every example already has at least beam_size finished beams.
            if all(len(finished[ex_idx]) >= beam_size for ex_idx in range(B)):
                break

        # Stragglers: any beam that never hit a boundary still contributes.
        for ex_idx in range(B):
            for _input_ids, gen, lp in active_beams[ex_idx]:
                if gen:
                    finished[ex_idx].append((gen, lp))

        # Decode + dedupe + return top-k step strings per example.
        results: list[list[str]] = []
        for ex_idx in range(B):
            finished[ex_idx].sort(key=lambda x: -x[1])
            seen: set[str] = set()
            top_steps: list[str] = []
            for gen, _lp in finished[ex_idx]:
                s = tok.decode_step(gen)
                if s and s not in seen:
                    seen.add(s)
                    top_steps.append(s)
                if len(top_steps) >= k:
                    break
            results.append(top_steps)
        return results

    @torch.no_grad()
    def complete_steps_batch(
        self,
        family_list: Sequence[str],
        partial_steps_list: Sequence[Sequence[str]],
        max_steps_list: Sequence[int],
    ) -> list[list[str]]:
        """Batched greedy completion in subword mode.

        Generates subword tokens one at a time across the active batch,
        accumulating full step strings at each <sep> boundary, stopping
        per example on <eos> or when max_steps_list[i] steps are emitted.
        Subword mode skips PAD/family masking (must be free to emit specials).
        """
        assert len(family_list) == len(partial_steps_list) == len(max_steps_list)
        B = len(family_list)
        outputs: list[list[str]] = [[] for _ in range(B)]
        if B == 0:
            return outputs

        tok = self._tokenizer
        if not hasattr(tok, "sep_id"):
            raise RuntimeError("complete_steps_batch requires a subword tokenizer")

        input_ids_state: list[list[int]] = []
        current_step_ids: list[list[int]] = [[] for _ in range(B)]
        max_steps = list(max_steps_list)
        for fam, partial in zip(family_list, partial_steps_list):
            _fid, prefix_ids = tok.encode_prefix(fam, list(partial))
            input_ids_state.append(list(prefix_ids))

        active = list(range(B))
        # Hard cap shared by single-example complete_steps: max_seq_len * 2 tokens.
        max_total_tokens = self._model.config.max_seq_len * 2

        for _ in range(max_total_tokens):
            if not active:
                break
            sub_fams = [family_list[i] for i in active]
            sub_prefs = [input_ids_state[i] for i in active]
            # Argmax over UNMASKED vocab — subword needs to be free to emit <sep>/<eos>.
            last_logits = self._last_logits_batch(sub_fams, sub_prefs)  # (N, V+1)
            next_ids = last_logits.argmax(dim=-1).tolist()

            new_active: list[int] = []
            for slot, i in enumerate(active):
                nid = int(next_ids[slot])
                input_ids_state[i].append(nid)
                if nid == tok.eos_id:
                    # Flush the in-progress step (if any) and finish.
                    if current_step_ids[i]:
                        s = tok.decode_step(current_step_ids[i])
                        if s:
                            outputs[i].append(s)
                    continue
                if nid == tok.sep_id:
                    s = tok.decode_step(current_step_ids[i])
                    if s:
                        outputs[i].append(s)
                    current_step_ids[i] = []
                    if len(outputs[i]) >= max_steps[i]:
                        continue
                else:
                    current_step_ids[i].append(nid)
                new_active.append(i)
            active = new_active
        return outputs

    @torch.no_grad()
    def log_prob_steps_batch(
        self,
        family_list: Sequence[str],
        steps_list: Sequence[Sequence[str]],
    ) -> list[float]:
        """Batched teacher-forced log-prob of step sequences (subword mode)."""
        assert len(family_list) == len(steps_list)
        if not family_list:
            return []
        tok = self._tokenizer
        if not hasattr(tok, "sep_id"):
            raise RuntimeError("log_prob_steps_batch requires a subword tokenizer")

        ids_list: list[list[int]] = []
        for fam, steps in zip(family_list, steps_list):
            if not steps:
                ids_list.append([])
                continue
            _fid, ids = tok.encode_sequence(fam, list(steps))
            ids_list.append(ids)
        return self.log_prob_batch(family_list, ids_list)
