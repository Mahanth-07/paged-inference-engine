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
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        q, k = rotary(positions, q, k)
        attn_out = self.attn(q, k, v, kv_cache, model_input)
        # reshape, not view: SDPA and the various kernels do not all return the same
        # stride layout, and a non-contiguous (tokens, heads, dim) tensor cannot be
        # viewed as (tokens, heads*dim). reshape is free when it is already contiguous.
        return self.o_proj(attn_out.reshape(num_tokens, self.q_size))


class LlamaMLP(nn.Module):
    """down(silu(gate(x)) * up(x)), with gate and up fused into one GEMM."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(silu_and_mul(self.gate_up_proj(x)))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, intermediate_size: int,
                 backend: str = "naive"):
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx, backend=backend)
        self.mlp = LlamaMLP(config.hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, residual, positions, rotary, kv_cache, model_input):
        # The (hidden, residual) pair threading is what lets the norm fuse the add.
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(hidden_states, positions, rotary, kv_cache, model_input)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):
    def __init__(self, config: ModelConfig, intermediate_size: int, backend: str = "naive",
                 rope_scaling: Optional[dict] = None):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, i, intermediate_size, backend=backend)
                for i in range(config.num_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary = RotaryEmbedding(
            head_dim=config.head_dim,
            max_position=config.max_model_len,
            base=config.rope_theta,
            rope_scaling=rope_scaling,
        )

    def forward(self, input_ids, positions, kv_caches, model_input):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, positions, self.rotary,
                kv_caches[i] if kv_caches else None, model_input,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig, intermediate_size: int, backend: str = "naive",
                 tie_word_embeddings: bool = True, rope_scaling: Optional[dict] = None):
        super().__init__()
        self.config = config
        self.tie_word_embeddings = tie_word_embeddings
        self.model = LlamaModel(config, intermediate_size, backend=backend,
                                rope_scaling=rope_scaling)
        if tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, positions, kv_caches, model_input):
        return self.model(input_ids, positions, kv_caches, model_input)

    def compute_logits(self, hidden_states, logits_indices=None):
        """Only compute logits where they are needed.

        A 128k-vocab lm_head over all 8192 batched tokens instead of over the ~64
        positions that actually need a token is a real and common waste: it is the
        single biggest GEMM in the step and most of it is thrown away.
        """
        if logits_indices is not None:
            hidden_states = hidden_states.index_select(0, logits_indices)
        weight = self.model.embed_tokens.weight if self.lm_head is None else self.lm_head.weight
        return F.linear(hidden_states, weight)

    # ---- weight loading ---------------------------------------------------

    # HF checkpoint name -> (our param name, fused-buffer slice index or None)
    _FUSED_QKV = {"q_proj": 0, "k_proj": 1, "v_proj": 2}
    _FUSED_MLP = {"gate_proj": 0, "up_proj": 1}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> dict[str, int]:
        """Load an HF Llama checkpoint, fusing q/k/v and gate/up on the way in.

        Returns {our_param_name: number_of_writes} so the caller can assert nothing was
        left at its random init and that every fused buffer got all of its pieces.
        A missing weight produces fluent garbage, which is the hardest failure to notice.
        """
        params = dict(self.named_parameters())
        loaded: dict[str, int] = {}
        cfg = self.config
        q_size, kv_size = cfg.num_heads * cfg.head_dim, cfg.num_kv_heads * cfg.head_dim
        qkv_offsets = [0, q_size, q_size + kv_size, q_size + 2 * kv_size]

        for name, tensor in weights:
            if name.endswith(".rotary_emb.inv_freq") or "rotary_emb" in name:
                continue
            if name == "lm_head.weight" and self.tie_word_embeddings:
                continue  # tied -- embed_tokens is the source of truth

            target, sl = name, None
            for hf_name, idx in self._FUSED_QKV.items():
                if f"self_attn.{hf_name}.weight" in name:
                    target = name.replace(f"self_attn.{hf_name}", "self_attn.qkv_proj")
                    sl = slice(qkv_offsets[idx], qkv_offsets[idx + 1])
                    break
            else:
                for hf_name, idx in self._FUSED_MLP.items():
                    if f"mlp.{hf_name}.weight" in name:
                        target = name.replace(f"mlp.{hf_name}", "mlp.gate_up_proj")
                        inter = params[target].shape[0] // 2
                        sl = slice(idx * inter, (idx + 1) * inter)
                        break

            if target not in params:
                continue
            param = params[target]
            src = tensor.to(dtype=param.dtype)
            if sl is None:
                if param.shape != src.shape:
                    raise ValueError(f"shape mismatch for {name}: {tuple(param.shape)} vs {tuple(src.shape)}")
                param.data.copy_(src)
            else:
                param.data[sl].copy_(src)
            loaded[target] = loaded.get(target, 0) + 1
        return loaded

    def expected_write_counts(self) -> dict[str, int]:
        """How many checkpoint tensors should land in each of our params.

        Fused buffers expect more than one: qkv_proj gets q, k and v; gate_up_proj gets
        gate and up. Compare against load_weights()'s return value at startup.
        """
        expected = {}
        for name in dict(self.named_parameters()):
            if name.endswith("qkv_proj.weight"):
                expected[name] = 3
