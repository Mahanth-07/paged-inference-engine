"""Attention backend microbenchmark and roofline check (GPU only).

    python -m bench.microbench_attention
    python -m bench.microbench_attention --roofline --context 8192 --batch 1

Decode attention is MEMORY BOUND: one token of query against a full sweep of the KV
cache. So the number that matters is not TFLOPs, it is achieved HBM bandwidth as a
fraction of peak. `--roofline` computes exactly that:

    bytes_moved = num_seqs * context_len * num_kv_heads * head_dim * 2 (K and V) * dtype
    achieved GB/s = bytes_moved / elapsed

A kernel at 80% of peak has essentially nothing left to give and the remaining gap to
vLLM is somewhere else. A kernel at 25% has a real problem -- usually uncoalesced
loads from a too-small page tile, or too few programs to saturate the SMs (which is
what SPLIT_K exists to fix).
"""

from __future__ import annotations

import argparse
import time

import torch

from paged_inference_engine.kernels.paged_attention_triton import (
    paged_attention_decode,
    paged_attention_prefill,
    reference_paged_attention_decode,
)


def make_fixture(num_seqs, context_len, num_heads, num_kv_heads, head_dim,
                 block_size, dtype, device, seed=0):
    """Blocks are SHUFFLED on purpose. A fixture with contiguous block ids passes even
    with a completely broken block-table lookup, which defeats the point."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    pages_per_seq = (context_len + block_size - 1) // block_size
    num_blocks = pages_per_seq * num_seqs + 8
    perm = torch.randperm(num_blocks, generator=g)[:pages_per_seq * num_seqs]
    block_tables = perm.view(num_seqs, pages_per_seq).to(torch.int32).to(device)

    k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim,
                          dtype=dtype, device=device)
    v_cache = torch.randn_like(k_cache)
    query = torch.randn(num_seqs, num_heads, head_dim, dtype=dtype, device=device)
    context_lens = torch.full((num_seqs,), context_len, dtype=torch.int32, device=device)
    return query, k_cache, v_cache, block_tables, context_lens


def timeit(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def peak_bandwidth(device) -> float:
    """Measured, not datasheet. A big device-to-device copy is the practical ceiling."""
    x = torch.empty(1 << 27, dtype=torch.float16, device=device)
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 30
    for _ in range(n):
