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
        y.copy_(x)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return 2 * x.numel() * 2 * n / dt / 1e9


def bench_decode(args):
    device, dtype = "cuda", torch.float16
    scale = args.head_dim ** -0.5
    peak = peak_bandwidth(device)
    print(f"measured peak HBM bandwidth: {peak:.0f} GB/s "
          f"({torch.cuda.get_device_name(0)})\n")

    hdr = (f"{'batch':>6} {'ctx':>7} {'backend':>18} {'us':>9} {'GB/s':>9} {'%peak':>7}")
    print(hdr)
    print("-" * len(hdr))

    for batch in args.batch:
        for ctx in args.context:
            q, kc, vc, bt, cl = make_fixture(
                batch, ctx, args.num_heads, args.num_kv_heads, args.head_dim,
                args.block_size, dtype, device)
            kv_bytes = batch * ctx * args.num_kv_heads * args.head_dim * 2 * 2

            cands = {
                "triton (v1)": lambda: paged_attention_decode(q, kc, vc, bt, cl, scale,
                                                              split_k=1),
                "triton (split-K)": lambda: paged_attention_decode(q, kc, vc, bt, cl, scale),
            }
            try:
                import flashinfer  # noqa: F401

                cands["flashinfer"] = _flashinfer_decode_fn(
                    q, kc, vc, bt, cl, args, scale, device, dtype)
            except ImportError:
                pass
            try:
                from flash_attn import flash_attn_with_kvcache

                cands["flash_attn"] = lambda: flash_attn_with_kvcache(
                    q.unsqueeze(1), kc, vc, cache_seqlens=cl, block_table=bt,
                    softmax_scale=scale, causal=True)
            except ImportError:
                pass
            if args.include_naive:
                cands["naive (oracle)"] = lambda: reference_paged_attention_decode(
                    q, kc, vc, bt, cl, scale)

            for name, fn in cands.items():
                try:
                    dt = timeit(fn, iters=args.iters)
                except Exception as e:  # noqa: BLE001
                    print(f"{batch:>6} {ctx:>7} {name:>18}  FAILED: {type(e).__name__}: {e}")
                    continue
                gbs = kv_bytes / dt / 1e9
                print(f"{batch:>6} {ctx:>7} {name:>18} {dt*1e6:>9.1f} {gbs:>9.0f} "
                      f"{100*gbs/peak:>6.1f}%")
        print()


def _flashinfer_decode_fn(q, kc, vc, bt, cl, args, scale, device, dtype):
    import flashinfer

    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
    pages = bt.shape[1]
    indptr = torch.arange(0, (bt.shape[0] + 1) * pages, pages,
                          dtype=torch.int32, device=device)
    indices = bt.reshape(-1).contiguous()
    last = torch.full((bt.shape[0],),
                      int(cl[0]) - (pages - 1) * args.block_size,
                      dtype=torch.int32, device=device)
    wrapper.plan(indptr, indices, last, args.num_heads, args.num_kv_heads,
                 args.head_dim, args.block_size, q_data_type=dtype)
    return lambda: wrapper.run(q, (kc, vc))


def bench_prefill(args):
    device, dtype = "cuda", torch.float16
    scale = args.head_dim ** -0.5
    print(f"{'seqs':>6} {'qlen':>7} {'backend':>18} {'ms':>9} {'TFLOP/s':>9}")
    print("-" * 52)
    for n in args.batch:
        for qlen in args.context:
            pages = (qlen + args.block_size - 1) // args.block_size
            num_blocks = pages * n + 8
            bt = torch.randperm(num_blocks)[:pages * n].view(n, pages).to(torch.int32).to(device)
            kc = torch.randn(num_blocks, args.block_size, args.num_kv_heads,
                             args.head_dim, dtype=dtype, device=device)
            vc = torch.randn_like(kc)
            q = torch.randn(n * qlen, args.num_heads, args.head_dim, dtype=dtype, device=device)
            cl = torch.full((n,), qlen, dtype=torch.int32, device=device)
            ql = torch.full((n,), qlen, dtype=torch.int32, device=device)
            indptr = torch.arange(0, (n + 1) * qlen, qlen, dtype=torch.int32, device=device)
            # causal: half the full q*k matrix
            flops = 2 * 2 * n * qlen * qlen * args.num_heads * args.head_dim / 2
            try:
                dt = timeit(lambda: paged_attention_prefill(
                    q, kc, vc, bt, cl, ql, indptr, scale), iters=args.iters)
                print(f"{n:>6} {qlen:>7} {'triton':>18} {dt*1e3:>9.3f} "
                      f"{flops/dt/1e12:>9.1f}")
            except Exception as e:  # noqa: BLE001
                print(f"{n:>6} {qlen:>7} {'triton':>18}  FAILED: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 64, 256])
    ap.add_argument("--context", type=int, nargs="+", default=[512, 4096, 16384])
    ap.add_argument("--num-heads", type=int, default=32)
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--include-naive", action="store_true",
                    help="Include the O(context) gather oracle. Slow; illustrative.")
    ap.add_argument("--prefill", action="store_true")
    ap.add_argument("--roofline", action="store_true", help="(implied; always reported)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU -- this is the part you rent a box for")
    if args.prefill:
