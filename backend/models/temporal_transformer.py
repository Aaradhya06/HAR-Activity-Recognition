"""
Temporal Transformer Module
============================
Implements a compact Transformer encoder for modelling temporal relationships
across a sequence of per-frame CNN feature embeddings.

Input:  (B, T, d_model) — sequence of spatial embeddings
Output: (B, d_model)    — pooled temporal representation

Architecture:
  1. Positional encoding (learnable or sinusoidal)
  2. N × Transformer encoder layers (multi-head self-attention + FFN)
  3. Temporal pooling (cls token or mean pooling)

The module also ships a lightweight LSTM/GRU fallback for environments
where the Transformer is too slow or where ablation experiments are needed.
"""

from __future__ import annotations

import math
import logging
from typing import Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

TemporalType = Literal["transformer", "lstm", "gru"]
PoolingType = Literal["cls", "mean", "last"]


# ── Positional Encodings ──────────────────────────────────────────────────────

class LearnablePositionalEncoding(nn.Module):
    """Learnable positional embeddings."""

    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        return self.dropout(x + self.pe(positions))


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encodings (Vaswani et al., 2017)."""

    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        x = x + self.pe[:, :T]  # type: ignore[index]
        return self.dropout(x)


# ── Transformer Temporal Module ───────────────────────────────────────────────

class TemporalTransformer(nn.Module):
    """
    Compact Transformer encoder for temporal modelling over CNN features.

    Args:
        input_dim: Dimensionality of input CNN features per frame.
        d_model: Internal Transformer dimension (may differ from input_dim).
        nhead: Number of attention heads (d_model must be divisible by nhead).
        num_layers: Number of Transformer encoder layers.
        dim_feedforward: FFN hidden dimension within each encoder layer.
        dropout: Dropout probability.
        max_len: Maximum sequence length (for positional encoding).
        pooling: Temporal pooling strategy — 'cls', 'mean', or 'last'.
        positional_encoding: 'learnable' or 'sinusoidal'.
    """

    def __init__(
        self,
        input_dim: int = 256,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 64,
        pooling: PoolingType = "mean",
        positional_encoding: Literal["learnable", "sinusoidal"] = "learnable",
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.d_model = d_model
        self.pooling = pooling

        # Project input features to d_model if needed
        if input_dim != d_model:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.LayerNorm(d_model),
            )
        else:
            self.input_proj = nn.Identity()

        # CLS token for classification pooling
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding
        seq_len = max_len + (1 if pooling == "cls" else 0)
        if positional_encoding == "learnable":
            self.pos_enc = LearnablePositionalEncoding(seq_len, d_model, dropout)
        else:
            self.pos_enc = SinusoidalPositionalEncoding(seq_len, d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        logger.info(
            "TemporalTransformer: d_model=%d, nhead=%d, layers=%d, pooling=%s",
            d_model, nhead, num_layers, pooling,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) — sequence of CNN feature embeddings.

        Returns:
            (B, d_model) — pooled temporal representation.
        """
        B = x.size(0)
        x = self.input_proj(x)                     # (B, T, d_model)

        if self.pooling == "cls":
            cls = self.cls_token.expand(B, -1, -1) # (B, 1, d_model)
            x = torch.cat([cls, x], dim=1)         # (B, T+1, d_model)

        x = self.pos_enc(x)                        # (B, T[+1], d_model)
        x = self.transformer(x)                    # (B, T[+1], d_model)

        if self.pooling == "cls":
            return x[:, 0]                         # CLS token output
        elif self.pooling == "mean":
            return x.mean(dim=1)
        else:  # "last"
            return x[:, -1]


# ── LSTM/GRU Fallback ─────────────────────────────────────────────────────────

class TemporalRNN(nn.Module):
    """
    LSTM or GRU-based temporal encoder — lightweight fallback / baseline.

    Used for:
      - Ablation study Experiment B (CNN + LSTM baseline)
      - Environments where Transformer overhead is prohibitive
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        rnn_type: Literal["lstm", "gru"] = "lstm",
    ) -> None:
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        multiplier = 2 if bidirectional else 1
        self.d_model = hidden_size * multiplier
        self.norm = nn.LayerNorm(self.d_model)

        logger.info(
            "TemporalRNN: type=%s, hidden=%d, layers=%d, bidirectional=%s",
            rnn_type, hidden_size, num_layers, bidirectional,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)

        Returns:
            (B, d_model) — last hidden state (bidirectional: concat fwd+bwd)
        """
        out, _ = self.rnn(x)   # (B, T, d_model)
        out = out[:, -1]       # take last timestep
        return self.norm(out)


# ── Factory function ──────────────────────────────────────────────────────────

def build_temporal_module(
    temporal_type: TemporalType,
    input_dim: int,
    cfg: dict,
) -> tuple[nn.Module, int]:
    """
    Factory to build the appropriate temporal module from config.

    Returns:
        (module, output_dim)
    """
    if temporal_type == "transformer":
        module = TemporalTransformer(
            input_dim=input_dim,
            d_model=cfg.get("transformer_d_model", 256),
            nhead=cfg.get("transformer_nhead", 8),
            num_layers=cfg.get("transformer_num_layers", 2),
            dim_feedforward=cfg.get("transformer_dim_feedforward", 512),
            dropout=cfg.get("transformer_dropout", 0.1),
            max_len=cfg.get("sequence_length", 32),
            pooling="mean",
        )
        output_dim = module.d_model
    elif temporal_type in ("lstm", "gru"):
        module = TemporalRNN(
            input_dim=input_dim,
            hidden_size=cfg.get("lstm_hidden_size", 256),
            num_layers=cfg.get("lstm_num_layers", 2),
            dropout=cfg.get("lstm_dropout", 0.3),
            bidirectional=cfg.get("lstm_bidirectional", False),
            rnn_type=temporal_type,
        )
        output_dim = module.d_model
    else:
        raise ValueError(f"Unknown temporal_type: {temporal_type!r}")

    return module, output_dim
