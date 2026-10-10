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
# references (pure torch -- run on the MacBook, and are the test oracle)
# ===========================================================================


def reference_paged_attention_decode(query, k_cache, v_cache, block_tables,
                                     context_lens, scale):
    """Pure-PyTorch paged decode. Slow, obviously correct, runs anywhere.

    query: (num_seqs, num_heads, head_dim)
    k_cache/v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
    """
    num_seqs, num_heads, head_dim = query.shape
    num_kv_heads = k_cache.shape[2]
    block_size = k_cache.shape[1]
    ratio = num_heads // num_kv_heads
    out = torch.empty_like(query)
    for i in range(num_seqs):
        ctx = int(context_lens[i])
        nb = (ctx + block_size - 1) // block_size
        blocks = block_tables[i, :nb]
        k = k_cache[blocks].reshape(-1, num_kv_heads, head_dim)[:ctx]
        v = v_cache[blocks].reshape(-1, num_kv_heads, head_dim)[:ctx]
        if ratio > 1:
            k = k.repeat_interleave(ratio, dim=1)
            v = v.repeat_interleave(ratio, dim=1)
        q = query[i].unsqueeze(1)                                # (H, 1, D)
        logits = (q.float() @ k.transpose(0, 1).float().transpose(1, 2)) * scale
        probs = logits.softmax(-1)
        out[i] = (probs @ v.transpose(0, 1).float()).squeeze(1).to(query.dtype)
    return out


def reference_paged_attention_prefill(query, k_cache, v_cache, block_tables,
                                      context_lens, query_lens, scale):
    """Pure-PyTorch paged prefill with bottom-right causal alignment.

    Bottom-right alignment is the chunked-prefill detail: a chunk's queries occupy
    absolute positions ctx - q_len .. ctx - 1, so query j may attend to key p iff
    p <= ctx - q_len + j. Top-left alignment (the naive `is_causal=True`) is wrong
    for every chunk after the first.
    """
    num_heads, head_dim = query.shape[1], query.shape[2]
    num_kv_heads = k_cache.shape[2]
    block_size = k_cache.shape[1]
    ratio = num_heads // num_kv_heads
    outs = []
    offset = 0
    for i, q_len in enumerate(int(x) for x in query_lens):
        ctx = int(context_lens[i])
        q = query[offset:offset + q_len]
        offset += q_len
        nb = (ctx + block_size - 1) // block_size
        blocks = block_tables[i, :nb]
        k = k_cache[blocks].reshape(-1, num_kv_heads, head_dim)[:ctx]
        v = v_cache[blocks].reshape(-1, num_kv_heads, head_dim)[:ctx]
        if ratio > 1:
            k = k.repeat_interleave(ratio, dim=1)
            v = v.repeat_interleave(ratio, dim=1)
        pos = torch.arange(ctx - q_len, ctx, device=q.device).unsqueeze(1)
        keys = torch.arange(ctx, device=q.device).unsqueeze(0)
        mask = keys <= pos
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0).float(),
            k.transpose(0, 1).unsqueeze(0).float(),
            v.transpose(0, 1).unsqueeze(0).float(),
            attn_mask=mask, scale=scale,
        )
        outs.append(o.squeeze(0).transpose(0, 1).to(query.dtype))
    return torch.cat(outs, dim=0)


# ===========================================================================
# kernels
# ===========================================================================

if HAS_TRITON:

    @triton.jit
    def _paged_attn_decode_kernel(
        out_ptr,             # (num_seqs, num_heads, head_dim)
        q_ptr,               # (num_seqs, num_heads, head_dim)
        k_cache_ptr,         # (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache_ptr,
        block_tables_ptr,    # (num_seqs, max_num_pages) int32
        context_lens_ptr,    # (num_seqs,) int32
        scale,
        stride_os, stride_oh,
        stride_qs, stride_qh,
        stride_kb, stride_kt, stride_kh,
        stride_bts,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        TILE_KV: tl.constexpr,
        GQA_RATIO: tl.constexpr,
        PAD_GQA: tl.constexpr,
    ):
        seq = tl.program_id(0)
        kv_head = tl.program_id(1)

        ctx_len = tl.load(context_lens_ptr + seq)
        if ctx_len <= 0:
            return

        rows = tl.arange(0, PAD_GQA)
        row_mask = rows < GQA_RATIO
        d = tl.arange(0, HEAD_DIM)

        # The GQA_RATIO query heads sharing this KV head, loaded once.
        q_off = seq * stride_qs + (kv_head * GQA_RATIO + rows)[:, None] * stride_qh + d[None, :]
        q = tl.load(q_ptr + q_off, mask=row_mask[:, None], other=0.0)

        m_i = tl.full([PAD_GQA], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([PAD_GQA], dtype=tl.float32)
        acc = tl.zeros([PAD_GQA, HEAD_DIM], dtype=tl.float32)

        tile_idx = tl.arange(0, TILE_KV)
        page_of = tile_idx // BLOCK_SIZE
        slot_of = tile_idx % BLOCK_SIZE

        num_tiles = tl.cdiv(ctx_len, TILE_KV)
        for t in range(0, num_tiles):
            tok = t * TILE_KV + tile_idx
            valid = tok < ctx_len
