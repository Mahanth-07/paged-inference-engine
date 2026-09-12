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

    Wavelengths shorter than the original context stay untouched, long ones are
    divided by `factor`, and there is a smooth ramp in between. Getting this wrong
    is invisible under 8k tokens and catastrophic above it.
    """
    low_wavelen = original_max_position / low_freq_factor
    high_wavelen = original_max_position / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    # long wavelengths (low frequency): divide by factor
    scaled = torch.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)
    # smooth ramp in the middle band
    smooth = (original_max_position / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed = (1 - smooth) * (inv_freq / factor) + smooth * inv_freq
    is_medium = (wavelen <= low_wavelen) & (wavelen >= high_wavelen)
    return torch.where(is_medium, smoothed, scaled)


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables, indexed by absolute position.

    Recomputing the tables every step is pure waste and shows up in a profile as a
    surprisingly fat elementwise kernel. Build once at startup, index thereafter.
    """

    def __init__(
        self,
        head_dim: int,
        max_position: int,
        base: float = 500000.0,
        rope_scaling: Optional[dict] = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
        )
        if rope_scaling and str(rope_scaling.get("rope_type", "")).lower() == "llama3":
            inv_freq = _llama3_scaled_inv_freq(
                inv_freq,
                float(rope_scaling.get("factor", 8.0)),
                float(rope_scaling.get("low_freq_factor", 1.0)),
                float(rope_scaling.get("high_freq_factor", 4.0)),
                int(rope_scaling.get("original_max_position_embeddings", 8192)),
            )
        t = torch.arange(max_position, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)                       # (max_pos, head_dim/2)
        self.register_buffer("cos_cache", freqs.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cache", freqs.sin().to(dtype), persistent=False)

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """query: (num_tokens, num_heads, head_dim), key: (num_tokens, num_kv_heads, head_dim)."""
        from ..kernels.fused import apply_rope

        cos = self.cos_cache[positions]     # (num_tokens, head_dim/2)
        sin = self.sin_cache[positions]
        return apply_rope(query, key, cos, sin)


# ---------------------------------------------------------------------------
# transformer blocks
# ---------------------------------------------------------------------------


class LlamaAttention(nn.Module):
    """qkv fused into one GEMM, then RoPE, then paged attention, then o_proj.

    Three separate small GEMMs for q/k/v is three kernel launches and three passes
    over the same input. On a 1B model at batch 1 that is measurable.
    """

    def __init__(self, config: ModelConfig, layer_idx: int, backend: str = "naive"):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = nn.Linear(config.hidden_size, self.q_size + 2 * self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=False)
        self.attn = PagedAttention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            scale=self.head_dim ** -0.5,
            backend=backend,
            layer_idx=layer_idx,
        )

    def forward(self, hidden_states, positions, rotary, kv_cache, model_input):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        num_tokens = hidden_states.shape[0]
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
