"""Fused elementwise kernels (M5), with torch fallbacks that run anywhere.

Every function here has two implementations: a plain-torch one that is the oracle
and runs on the MacBook, and a Triton one used on CUDA. `tests/test_kernels.py`
asserts they agree, so the fallback is not just dead code -- it is the reference.

Why these ops are worth fusing at all: none of them is expensive, but each launches
2-6 kernels and makes as many passes over the hidden state, twice per decoder layer.
On a 1B model at batch size 1 the decode step is launch-bound, so removing launches
is removing latency. Measure it with `bench/microbench_kernels.py`.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # laptop, or any non-CUDA box
    HAS_TRITON = False
    triton = None
    tl = None

_DISABLE = os.environ.get("PAGED_INFERENCE_ENGINE_DISABLE_TRITON", "0") == "1"


def _use_triton(x: torch.Tensor) -> bool:
    return HAS_TRITON and not _DISABLE and x.is_cuda


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


def rms_norm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x32 = x.float()
    var = x32.pow(2).mean(-1, keepdim=True)
    return (x32 * torch.rsqrt(var + eps)).to(dtype) * weight


def fused_add_rms_norm_torch(x, residual, weight, eps):
    residual = residual + x
    return rms_norm_torch(residual, weight, eps), residual


if HAS_TRITON:

    @triton.jit
    def _rms_norm_kernel(
        out_ptr, x_ptr, w_ptr, eps, n_cols,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * n_cols + cols, (x * rstd) * w, mask=mask)

    @triton.jit
    def _fused_add_rms_norm_kernel(
        out_ptr, x_ptr, res_ptr, w_ptr, eps, n_cols,
        BLOCK: tl.constexpr,
    ):
        """Adds the residual, writes it back, and normalises -- one pass, one launch.

        Writing the updated residual in place is what lets the next layer read it
        without another kernel; that in-place write is the whole point of the fusion.
        """
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(res_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        r = r + x
        tl.store(res_ptr + row * n_cols + cols, r, mask=mask)
        var = tl.sum(r * r, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * n_cols + cols, (r * rstd) * w, mask=mask)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if not _use_triton(x):
        return rms_norm_torch(x, weight, eps)
    x2 = x.contiguous().view(-1, x.shape[-1])
    n_cols = x2.shape[-1]
    out = torch.empty_like(x2)
    _rms_norm_kernel[(x2.shape[0],)](
        out, x2, weight, eps, n_cols, BLOCK=triton.next_power_of_2(n_cols), num_warps=8
    )
    return out.view(x.shape)


def fused_add_rms_norm(x, residual, weight, eps):
    if not _use_triton(x):
        return fused_add_rms_norm_torch(x, residual, weight, eps)
    x2 = x.contiguous().view(-1, x.shape[-1])
    res2 = residual.contiguous().view(-1, x.shape[-1])
    n_cols = x2.shape[-1]
    out = torch.empty_like(x2)
    _fused_add_rms_norm_kernel[(x2.shape[0],)](
        out, x2, res2, weight, eps, n_cols,
        BLOCK=triton.next_power_of_2(n_cols), num_warps=8,
    )
    return out.view(x.shape), res2.view(residual.shape)


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------


def silu_and_mul_torch(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


if HAS_TRITON:

    @triton.jit
    def _silu_and_mul_kernel(out_ptr, x_ptr, d, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        for start in range(0, d, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < d
            g = tl.load(x_ptr + row * 2 * d + cols, mask=mask, other=0.0).to(tl.float32)
            u = tl.load(x_ptr + row * 2 * d + d + cols, mask=mask, other=0.0).to(tl.float32)
            silu = g / (1.0 + tl.exp(-g))
            tl.store(out_ptr + row * d + cols, silu * u, mask=mask)


def silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
    """SwiGLU. torch needs a chunk (a view + two reads) plus silu plus mul: 3 kernels
    and 3 passes. The fused version is one pass over 2*d and one write of d."""
    if not _use_triton(gate_up):
        return silu_and_mul_torch(gate_up)
    x = gate_up.contiguous().view(-1, gate_up.shape[-1])
    d = x.shape[-1] // 2
    out = torch.empty(x.shape[0], d, dtype=gate_up.dtype, device=gate_up.device)
    _silu_and_mul_kernel[(x.shape[0],)](out, x, d, BLOCK=1024, num_warps=4)
    return out.view(*gate_up.shape[:-1], d)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def apply_rope_torch(query, key, cos, sin):
    """NeoX-style half rotation, which is what Llama uses.

    query: (num_tokens, num_heads, head_dim); cos/sin: (num_tokens, head_dim/2).
    """
    def rotate(x):
        x1, x2 = x.chunk(2, dim=-1)
        c = cos.unsqueeze(1).to(x.dtype)
        s = sin.unsqueeze(1).to(x.dtype)
        return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)

    return rotate(query), rotate(key)


if HAS_TRITON:

