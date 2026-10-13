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
            page_idx = t * (TILE_KV // BLOCK_SIZE) + page_of
            phys = tl.load(block_tables_ptr + seq * stride_bts + page_idx,
                           mask=valid, other=0)

            base = phys * stride_kb + slot_of * stride_kt + kv_head * stride_kh
            # K loaded transposed -> (HEAD_DIM, TILE_KV), ready for tl.dot
            k_t = tl.load(k_cache_ptr + base[None, :] + d[:, None],
                          mask=valid[None, :], other=0.0)
            qk = tl.dot(q, k_t) * scale
            qk = tl.where(valid[None, :], qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])

            v_tile = tl.load(v_cache_ptr + base[:, None] + d[None, :],
                             mask=valid[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new

        acc = acc / l_i[:, None]
        o_off = seq * stride_os + (kv_head * GQA_RATIO + rows)[:, None] * stride_oh + d[None, :]
        tl.store(out_ptr + o_off, acc.to(out_ptr.dtype.element_ty), mask=row_mask[:, None])

    # -----------------------------------------------------------------------

    @triton.jit
    def _paged_attn_decode_split_kernel(
        acc_ptr,             # (num_seqs, num_heads, SPLIT_K, head_dim) fp32
        lse_m_ptr,           # (num_seqs, num_heads, SPLIT_K) fp32  running max
        lse_l_ptr,           # (num_seqs, num_heads, SPLIT_K) fp32  running sum
        q_ptr, k_cache_ptr, v_cache_ptr,
        block_tables_ptr, context_lens_ptr,
        scale,
        stride_as, stride_ah, stride_ak,
        stride_ls, stride_lh,
        stride_qs, stride_qh,
        stride_kb, stride_kt, stride_kh,
        stride_bts,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        TILE_KV: tl.constexpr,
        GQA_RATIO: tl.constexpr,
        PAD_GQA: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        """Flash-decoding partial pass: same math, but each program owns a slice of
        the KV tiles. Turns a (num_seqs, num_kv_heads) grid into a
        (num_seqs, num_kv_heads, SPLIT_K) one, which is the difference between
        occupying 8 SMs and occupying all of them at batch size 1."""
        seq = tl.program_id(0)
        kv_head = tl.program_id(1)
        split = tl.program_id(2)

        ctx_len = tl.load(context_lens_ptr + seq)
        rows = tl.arange(0, PAD_GQA)
        row_mask = rows < GQA_RATIO
        d = tl.arange(0, HEAD_DIM)
        heads = kv_head * GQA_RATIO + rows

        num_tiles = tl.cdiv(ctx_len, TILE_KV)
        tiles_per_split = tl.cdiv(num_tiles, SPLIT_K)
        t_begin = split * tiles_per_split
        t_end = tl.minimum(t_begin + tiles_per_split, num_tiles)

        m_i = tl.full([PAD_GQA], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([PAD_GQA], dtype=tl.float32)
        acc = tl.zeros([PAD_GQA, HEAD_DIM], dtype=tl.float32)

        if t_begin < t_end:
            q_off = seq * stride_qs + heads[:, None] * stride_qh + d[None, :]
            q = tl.load(q_ptr + q_off, mask=row_mask[:, None], other=0.0)

            tile_idx = tl.arange(0, TILE_KV)
            page_of = tile_idx // BLOCK_SIZE
            slot_of = tile_idx % BLOCK_SIZE

            for t in range(t_begin, t_end):
                tok = t * TILE_KV + tile_idx
                valid = tok < ctx_len
                page_idx = t * (TILE_KV // BLOCK_SIZE) + page_of
                phys = tl.load(block_tables_ptr + seq * stride_bts + page_idx,
                               mask=valid, other=0)
                base = phys * stride_kb + slot_of * stride_kt + kv_head * stride_kh
                k_t = tl.load(k_cache_ptr + base[None, :] + d[:, None],
                              mask=valid[None, :], other=0.0)
                qk = tl.dot(q, k_t) * scale
                qk = tl.where(valid[None, :], qk, float("-inf"))
                m_new = tl.maximum(m_i, tl.max(qk, 1))
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(qk - m_new[:, None])
                v_tile = tl.load(v_cache_ptr + base[:, None] + d[None, :],
                                 mask=valid[:, None], other=0.0)
                acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
                l_i = l_i * alpha + tl.sum(p, 1)
                m_i = m_new

        a_off = seq * stride_as + heads[:, None] * stride_ah + split * stride_ak + d[None, :]
        tl.store(acc_ptr + a_off, acc, mask=row_mask[:, None])
        l_off = seq * stride_ls + heads * stride_lh + split
        tl.store(lse_m_ptr + l_off, m_i, mask=row_mask)
        tl.store(lse_l_ptr + l_off, l_i, mask=row_mask)

    @triton.jit
    def _combine_kernel(
        out_ptr, acc_ptr, lse_m_ptr, lse_l_ptr,
        stride_os, stride_oh,
        stride_as, stride_ah, stride_ak,
        stride_ls, stride_lh,
        HEAD_DIM: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        """Merge SPLIT_K partial softmaxes. Standard log-sum-exp rescale:
        each split contributes exp(m_k - m) * acc_k, weighted sum over exp(m_k - m)*l_k.
        Splits that got no tiles have l=0 and acc=0 and drop out on their own."""
        seq = tl.program_id(0)
        head = tl.program_id(1)
        d = tl.arange(0, HEAD_DIM)
        splits = tl.arange(0, SPLIT_K)

        l_off = seq * stride_ls + head * stride_lh + splits
        m_k = tl.load(lse_m_ptr + l_off)
        l_k = tl.load(lse_l_ptr + l_off)
        m = tl.max(m_k, 0)
        w = tl.exp(m_k - m)
        denom = tl.sum(w * l_k, 0)

        a_off = seq * stride_as + head * stride_ah + splits[:, None] * stride_ak + d[None, :]
        a = tl.load(acc_ptr + a_off)
        num = tl.sum(a * w[:, None], 0)
        tl.store(out_ptr + seq * stride_os + head * stride_oh + d,
                 (num / denom).to(out_ptr.dtype.element_ty))

    # -----------------------------------------------------------------------

    @triton.jit
    def _paged_prefill_kernel(
        out_ptr,             # (num_q_tokens, num_heads, head_dim)
        q_ptr,
        k_cache_ptr, v_cache_ptr,
        block_tables_ptr, context_lens_ptr, query_lens_ptr, qo_indptr_ptr,
        scale,
        stride_ot, stride_oh,
        stride_qt, stride_qh,
        stride_kb, stride_kt, stride_kh,
        stride_bts,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        TILE_KV: tl.constexpr,
        BLOCK_M: tl.constexpr,
        GQA_RATIO: tl.constexpr,
    ):
        """FlashAttention-2 forward, K/V fetched through the block table.

        Causal masking is BOTTOM-RIGHT aligned: query row m of a chunk sits at absolute
        position (ctx_len - q_len + m). Under chunked prefill the chunk does not start
        at position 0, so top-left `is_causal` would let early queries see the future
        and hide the rest of their own context."""
        m_block = tl.program_id(0)
        seq = tl.program_id(1)
        head = tl.program_id(2)

        q_len = tl.load(query_lens_ptr + seq)
        if m_block * BLOCK_M >= q_len:
            return
        ctx_len = tl.load(context_lens_ptr + seq)
        q_start = tl.load(qo_indptr_ptr + seq)
        kv_head = head // GQA_RATIO

        offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        m_valid = offs_m < q_len
        abs_pos = ctx_len - q_len + offs_m          # absolute position of each query row
        d = tl.arange(0, HEAD_DIM)

        q = tl.load(
            q_ptr + (q_start + offs_m)[:, None] * stride_qt + head * stride_qh + d[None, :],
            mask=m_valid[:, None], other=0.0,
        )

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        tile_idx = tl.arange(0, TILE_KV)
        page_of = tile_idx // BLOCK_SIZE
        slot_of = tile_idx % BLOCK_SIZE

        # Causality bounds the KV we need: nothing past the last query's position.
        max_key = tl.minimum(ctx_len, ctx_len - q_len + (m_block + 1) * BLOCK_M)
        num_tiles = tl.cdiv(max_key, TILE_KV)

        for t in range(0, num_tiles):
            tok = t * TILE_KV + tile_idx
            valid = tok < ctx_len
            page_idx = t * (TILE_KV // BLOCK_SIZE) + page_of
            phys = tl.load(block_tables_ptr + seq * stride_bts + page_idx,
                           mask=valid, other=0)
            base = phys * stride_kb + slot_of * stride_kt + kv_head * stride_kh
            k_t = tl.load(k_cache_ptr + base[None, :] + d[:, None],
                          mask=valid[None, :], other=0.0)
            qk = tl.dot(q, k_t) * scale
            keep = valid[None, :] & (tok[None, :] <= abs_pos[:, None])
            qk = tl.where(keep, qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            # A row whose every key is masked stays at -inf; force alpha/p to 0 there
            # instead of producing inf-inf = nan.
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(tl.where(m_i == float("-inf"), 0.0, m_i) - m_safe)
            alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
            p = tl.exp(qk - m_safe[:, None])
            p = tl.where(keep, p, 0.0)

            v_tile = tl.load(v_cache_ptr + base[:, None] + d[None, :],
                             mask=valid[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new

        acc = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
        tl.store(
            out_ptr + (q_start + offs_m)[:, None] * stride_ot + head * stride_oh + d[None, :],
            acc.to(out_ptr.dtype.element_ty), mask=m_valid[:, None],
        )


# ===========================================================================
# host-side launchers
# ===========================================================================


def _require_triton():
    if not HAS_TRITON or _DISABLE:
        raise RuntimeError(
            "Triton is unavailable (you are probably on the Mac). Use backend='naive'."
        )


def choose_split_k(num_seqs: int, num_kv_heads: int, max_context: int,
                   num_sms: int = 108, tile_kv: int = 64) -> int:
    """How many ways to split the KV sweep.

    The v1 grid is num_seqs * num_kv_heads programs. When that is well under the SM
    count the GPU is mostly idle, so split until we fill it -- but never into slices
    smaller than one tile, which would just add combine overhead for no parallelism.
    """
    base = max(1, num_seqs * num_kv_heads)
    if base >= num_sms:
        return 1
    max_useful = max(1, (max_context + tile_kv - 1) // tile_kv)
    want = min((num_sms + base - 1) // base, max_useful, 32)
    # Round DOWN to a power of two last, not first: _combine_kernel indexes the splits
    # with tl.arange(0, SPLIT_K), which requires a power of two. Clamping after the
    # rounding can hand back e.g. 3 and the combine kernel fails to compile.
    return 1 << int(math.floor(math.log2(max(1, want))))


def paged_attention_decode(query, k_cache, v_cache, block_tables, context_lens,
                           scale, split_k: Optional[int] = None, out=None,
                           max_context: Optional[int] = None):
    """Paged decode attention.

    query:        (num_seqs, num_heads, head_dim)
    k/v_cache:    (num_blocks, block_size, num_kv_heads, head_dim)
    block_tables: (num_seqs, max_num_pages) int32
    context_lens: (num_seqs,) int32
    max_context:  host-side max(context_lens). Pass it! The caller already knows it --
                  it built context_lens from a Python list -- and reading it off the
                  device here would be a full sync, once per LAYER, every step.
    """
    _require_triton()
    num_seqs, num_heads, head_dim = query.shape
    block_size, num_kv_heads = k_cache.shape[1], k_cache.shape[2]
    assert num_heads % num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads"
    assert k_cache.shape[3] == head_dim and v_cache.shape == k_cache.shape
    assert block_tables.shape[0] == num_seqs and context_lens.shape[0] == num_seqs
    assert block_tables.dtype == torch.int32 and context_lens.dtype == torch.int32
    gqa = num_heads // num_kv_heads
    pad_gqa = max(MIN_DOT, triton.next_power_of_2(gqa))
    tile_kv = _pick_tile_kv(block_size)
    if out is None:
        out = torch.empty_like(query)

    if split_k is None:
        props = torch.cuda.get_device_properties(query.device)
        if max_context is None:
            max_context = int(context_lens.max())   # sync; avoid by passing it in
        split_k = choose_split_k(num_seqs, num_kv_heads, max_context,
                                 props.multi_processor_count, tile_kv)

    common = dict(
        HEAD_DIM=head_dim, BLOCK_SIZE=block_size, TILE_KV=tile_kv,
        GQA_RATIO=gqa, PAD_GQA=pad_gqa, num_warps=4, num_stages=2,
    )

    if split_k <= 1:
        _paged_attn_decode_kernel[(num_seqs, num_kv_heads)](
            out, query, k_cache, v_cache, block_tables, context_lens, scale,
            out.stride(0), out.stride(1),
            query.stride(0), query.stride(1),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            block_tables.stride(0),
            **common,
        )
        return out

    acc = torch.empty(num_seqs, num_heads, split_k, head_dim,
                      dtype=torch.float32, device=query.device)
    lse_m = torch.full((num_seqs, num_heads, split_k), float("-inf"),
                       dtype=torch.float32, device=query.device)
    lse_l = torch.zeros((num_seqs, num_heads, split_k),
                        dtype=torch.float32, device=query.device)
    _paged_attn_decode_split_kernel[(num_seqs, num_kv_heads, split_k)](
        acc, lse_m, lse_l, query, k_cache, v_cache, block_tables, context_lens, scale,
        acc.stride(0), acc.stride(1), acc.stride(2),
        lse_m.stride(0), lse_m.stride(1),
        query.stride(0), query.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        block_tables.stride(0),
        SPLIT_K=split_k, **common,
    )
    _combine_kernel[(num_seqs, num_heads)](
        out, acc, lse_m, lse_l,
        out.stride(0), out.stride(1),
        acc.stride(0), acc.stride(1), acc.stride(2),
        lse_m.stride(0), lse_m.stride(1),
        HEAD_DIM=head_dim, SPLIT_K=split_k, num_warps=4,
    )
    return out


def paged_attention_prefill(query, k_cache, v_cache, block_tables, context_lens,
                            query_lens, qo_indptr, scale, block_m: int = 64, out=None,
                            max_query_len: Optional[int] = None):
    """Paged (chunked) prefill attention.

    query:   (num_q_tokens, num_heads, head_dim), sequences concatenated
    qo_indptr: (num_seqs + 1,) int32 cumulative query lengths
    context_lens[i]: total KV length for sequence i AFTER this chunk is written
    """
    _require_triton()
    num_tokens, num_heads, head_dim = query.shape
    block_size, num_kv_heads = k_cache.shape[1], k_cache.shape[2]
    num_seqs = block_tables.shape[0]
    assert query_lens.shape[0] == num_seqs and context_lens.shape[0] == num_seqs
    assert qo_indptr.shape[0] == num_seqs + 1
    assert block_tables.dtype == torch.int32
    gqa = num_heads // num_kv_heads
    tile_kv = _pick_tile_kv(block_size)
    if out is None:
        out = torch.empty_like(query)
