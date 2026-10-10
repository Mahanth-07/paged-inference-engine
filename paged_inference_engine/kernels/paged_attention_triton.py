"""Our own paged-attention kernels. This is what makes paged-inference-engine a kernels project.

Why decode attention specifically:
  - Decode has query length 1, so there is no big GEMM -- it is a pure memory sweep
    over the KV cache. Arithmetic intensity is roughly the batch size.
  - That makes it MEMORY BOUND. The optimisation target is percent of peak HBM
    bandwidth, not TFLOPs (4090 ~1008 GB/s, A100 80GB SXM ~2039 GB/s, H100 SXM
    ~3350 GB/s). `bench/microbench_attention.py --roofline` reports the achieved
    fraction; that number is the honest measure of this kernel.
  - Paging is what makes it non-trivial: one sequence's K/V is scattered across
    arbitrary physical blocks, so you cannot stride through memory -- you gather
    block ids from the block table and loop.

Three kernels here:
  1. _paged_attn_decode_kernel        one program per (seq, kv_head). The GQA_RATIO
                                      query heads that share a KV head are processed
                                      together, so their shared K/V is loaded ONCE
                                      instead of GQA_RATIO times. On Llama-3.2-1B
                                      (ratio 4) that is a 4x cut in KV traffic, and
                                      KV traffic is the whole cost.
  2. _paged_attn_decode_split_kernel  flash-decoding: split the sequence's KV across
     + _combine_kernel                SPLIT_K programs, each producing a partial
                                      (acc, m, l), then merge. At batch 1 the v1 grid
                                      is (1, num_kv_heads) = 8 programs on a 108-SM
                                      GPU -- 93% of the machine idle. This fixes that,
                                      and it is why long-context low-batch decode is
                                      fast at all.
  3. _paged_prefill_kernel            FlashAttention-2 forward whose K/V comes through
                                      the block table, with bottom-right causal
                                      alignment so chunked prefill is correct.

The KV tile is decoupled from the page size: a tile gathers `TILE_KV // block_size`
pages at once. That is what lets block_size=8 work (a 8-wide tl.dot is illegal) and
what makes block_size=16 coalesce properly instead of issuing 16-element loads.

Correctness gotchas, all of them in tests/test_paged_attention.py:
  - the last page is partial: mask positions >= context_len or you attend to garbage
  - GQA: query head h reads KV head h // (num_heads // num_kv_heads)
  - accumulate in fp32, always -- fp16 accumulation drifts with context length
  - context_len not a multiple of block_size is THE most common bug source
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # laptop
    HAS_TRITON = False
    triton = None

    class _TL:  # so the @triton.jit block below can be skipped cleanly
        constexpr = object

    tl = _TL()

_DISABLE = os.environ.get("PAGED_INFERENCE_ENGINE_DISABLE_TRITON", "0") == "1"

# Minimum tl.dot tile. Query rows are padded up to this even when GQA_RATIO is 1;
# the wasted FLOPs are free because the kernel is bandwidth-bound anyway.
MIN_DOT = 16


def _pick_tile_kv(block_size: int, target: int = 64) -> int:
    """Gather several pages per tile so loads coalesce and tl.dot stays legal.

    Must be an exact multiple of block_size (the tile indexes pages by integer
    division) and at least MIN_DOT wide (tl.dot rejects a dimension under 16, which is
    what would otherwise rule out block_size=8).
    """
    pages = max(1, -(-target // block_size))     # ceil
    return pages * block_size


# ===========================================================================
