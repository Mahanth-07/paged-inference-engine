"""Minimal Llama 3.x, written from scratch.

Architecture:
    embed_tokens
    N x DecoderLayer:
        RMSNorm -> Attention(qkv proj, RoPE, PagedAttention, o proj) -> residual
        RMSNorm -> MLP(gate/up fused, SwiGLU, down)                  -> residual
    RMSNorm
    lm_head   (tied to embed_tokens on the 1B model)

Things that silently corrupt output, all handled below:
  - RoPE theta is 500000.0 on Llama 3, not 10000.0.
  - Llama 3.1/3.2 use a piecewise frequency rescale ("llama3" rope_scaling) that only
    matters above 8k but is cheap to get right.
  - Position ids during chunked prefill are ABSOLUTE, not chunk-relative.
  - RMSNorm accumulates in fp32 then casts back, even in a bf16 model.
  - q/k/v and gate/up are fused into single GEMMs at load time; the checkpoint stores
    them separately, so the loader has to slice into the fused buffer.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..kernels.fused import fused_add_rms_norm, rms_norm, silu_and_mul
from ..layers.attention import PagedAttention


# ---------------------------------------------------------------------------
# norms
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """x * rsqrt(mean(x^2) + eps) * weight, accumulated in fp32.

    Runs 2N times per forward pass, so M5 fuses the residual add into the same
    kernel (see kernels/fused.py) -- that halves the number of passes over the
    hidden state in the decoder block.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
        if residual is None:
            return rms_norm(x, self.weight, self.eps)
        return fused_add_rms_norm(x, residual, self.weight, self.eps)


# ---------------------------------------------------------------------------
# rotary embedding
# ---------------------------------------------------------------------------


def _llama3_scaled_inv_freq(
    inv_freq: torch.Tensor,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position: int,
) -> torch.Tensor:
    """The Llama 3.1/3.2 piecewise RoPE rescale.

