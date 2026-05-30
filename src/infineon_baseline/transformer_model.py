"""Small GPT-style causal language model for next-step prediction.

Architecture:
- Token embeddings: initialised from STStepEmbedder.vectors (vocab_size x d_model).
- Family conditioning: learnable family embedding (n_families x d_model) summed
  into every position of the input (simpler than a special prefix token).
- Positional embeddings: learned (max_seq_len x d_model).
- n_layers stacked TransformerEncoderLayer blocks with causal mask.
- LM head: weight-tied linear projection to vocab_size logits (no softmax).

Default arch matches all-MiniLM-L6-v2 dimension (384):
  d_model=384, n_heads=6, n_layers=4, ff_dim=1536, dropout=0.1, max_seq_len=256
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    vocab_size: int = 0            # set from tokenizer at construction time
    n_families: int = 0            # set from tokenizer at construction time
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 4
    ff_dim: int = 1536
    dropout: float = 0.1
    max_seq_len: int = 256
    freeze_embeddings: bool = False
    pad_id: int = 0                # id reserved for PAD (set to vocab_size slot)


class TransformerLM(nn.Module):
    """GPT-style causal LM with family conditioning."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        V = config.vocab_size
        D = config.d_model

        # Token embedding table — rows will be overwritten from ST vectors.
        # V+1 slots: indices 0..V-1 are real steps, index V is PAD.
        self.token_emb = nn.Embedding(V + 1, D, padding_idx=V)
        # Family embedding (summed into every position).
        self.family_emb = nn.Embedding(config.n_families, D)
        # Learned positional embeddings.
        self.pos_emb = nn.Embedding(config.max_seq_len, D)
        self.emb_drop = nn.Dropout(config.dropout)

        # Transformer stack — nn.TransformerEncoder with causal mask applied in forward.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # LM head — weight-tied with token embedding table (halves parameters).
        self.lm_head = nn.Linear(D, V + 1, bias=False)
        self.lm_head.weight = self.token_emb.weight   # weight tying

        self._init_weights()
        if config.freeze_embeddings:
            self.token_emb.weight.requires_grad_(False)

    def _init_weights(self) -> None:
        nn.init.normal_(self.family_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    @classmethod
    def from_tokenizer_and_embedder(
        cls,
        tokenizer,
        embedder,
        **arch_kwargs,
    ) -> "TransformerLM":
        """Construct from a fitted Tokenizer and STStepEmbedder.

        The token embedding table is initialised from embedder.vectors.
        arch_kwargs override TransformerConfig defaults (d_model, n_heads, etc.).
        """
        vocab_size = len(tokenizer.id_to_step)
        n_families = len(tokenizer.family_to_id)
        d_model = arch_kwargs.pop("d_model", 384)

        config = TransformerConfig(
            vocab_size=vocab_size,
            n_families=n_families,
            d_model=d_model,
            pad_id=vocab_size,   # PAD uses the extra slot beyond the vocab
            **arch_kwargs,
        )
        model = cls(config)

        # Initialise token embeddings from the ST embedder vectors.
        vecs = torch.from_numpy(embedder.vectors.astype(np.float32))  # (vocab_size, D_emb)
        if vecs.shape[1] != d_model:
            # Project if embedding dim differs from d_model (e.g. when d_model overridden).
            proj = nn.Linear(vecs.shape[1], d_model, bias=False)
            with torch.no_grad():
                vecs = proj(vecs)
        with torch.no_grad():
            model.token_emb.weight[:vocab_size] = vecs

        return model

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask (True = ignore) for causal self-attention."""
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(
        self,
        family_ids: torch.Tensor,                        # (B,)
        step_ids: torch.Tensor,                          # (B, T)
        key_padding_mask: Optional[torch.Tensor] = None, # (B, T) True = PAD position
    ) -> torch.Tensor:                                   # (B, T, V+1)
        B, T = step_ids.shape
        device = step_ids.device

        # Build embeddings: token + position + family.
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # (B, T)
        tok = self.token_emb(step_ids)           # (B, T, D)
        pos = self.pos_emb(positions)            # (B, T, D)
        fam = self.family_emb(family_ids)        # (B, D)
        x = self.emb_drop(tok + pos + fam.unsqueeze(1))  # broadcast family across T

        causal = self._causal_mask(T, device)
        x = self.transformer(x, mask=causal, src_key_padding_mask=key_padding_mask,
                             is_causal=True)
        logits = self.lm_head(x)   # (B, T, V+1)
        return logits

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
