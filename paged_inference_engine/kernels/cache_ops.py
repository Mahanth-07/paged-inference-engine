"""reshape_and_cache, fused.

The torch version (`cache.view(-1, H, D).index_copy_(0, slots, k)`) launches several
kernels, materialises an index tensor, and makes two passes -- once for K and once
for V. It runs once per layer per step, so on a 1B model with 16 layers that is 32
scatter launches per decode step, against a step that is only a few milliseconds.
The fused version writes K and V in a single kernel with one pass over the slots.

Microbenchmark both: `python -m bench.microbench_kernels --op reshape_and_cache`.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None

_DISABLE = os.environ.get("PAGED_INFERENCE_ENGINE_DISABLE_TRITON", "0") == "1"


def use_triton_cache_op(x: torch.Tensor) -> bool:
    return HAS_TRITON and not _DISABLE and x.is_cuda


if HAS_TRITON:

    @triton.jit
    def _reshape_and_cache_kernel(
        k_ptr, v_ptr, k_cache_ptr, v_cache_ptr, slot_ptr,
        stride_kt, stride_kh,
        stride_ct, stride_ch,
        num_kv_heads,
        HEAD_DIM: tl.constexpr,
        HEADS_PER_BLOCK: tl.constexpr,
    ):
        """One program per token; writes all KV heads for that token, K and V together.

        A slot of -1 means "do not write" -- that is how CUDA-graph padding rows are
        neutralised without a host-side branch (see ModelRunner.capture_cuda_graphs).
        """
        tok = tl.program_id(0)
        slot = tl.load(slot_ptr + tok)
        if slot < 0:
            return

        d = tl.arange(0, HEAD_DIM)
        for h_start in range(0, num_kv_heads, HEADS_PER_BLOCK):
            h = h_start + tl.arange(0, HEADS_PER_BLOCK)
            h_mask = h < num_kv_heads
            src = tok * stride_kt + h[:, None] * stride_kh + d[None, :]
            dst = slot * stride_ct + h[:, None] * stride_ch + d[None, :]
            mask = h_mask[:, None]
            k = tl.load(k_ptr + src, mask=mask, other=0.0)
            v = tl.load(v_ptr + src, mask=mask, other=0.0)
            tl.store(k_cache_ptr + dst, k, mask=mask)
            tl.store(v_cache_ptr + dst, v, mask=mask)


def reshape_and_cache_triton(key, value, k_cache, v_cache, slot_mapping) -> None:
    num_tokens, num_kv_heads, head_dim = key.shape
    if num_tokens == 0:
        return
    # k_cache: (num_blocks, block_size, num_kv_heads, head_dim) -> flatten the first two
    kc = k_cache.view(-1, num_kv_heads, head_dim)
    vc = v_cache.view(-1, num_kv_heads, head_dim)
    _reshape_and_cache_kernel[(num_tokens,)](
        key, value, kc, vc, slot_mapping,
        key.stride(0), key.stride(1),
        kc.stride(0), kc.stride(1),
        num_kv_heads,
        HEAD_DIM=head_dim,
        HEADS_PER_BLOCK=min(triton.next_power_of_2(num_kv_heads), 8),
        num_warps=4,
    )
