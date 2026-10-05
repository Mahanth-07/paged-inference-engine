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

