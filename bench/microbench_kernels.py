"""Fused-kernel microbenchmarks: is the Triton version actually worth it? (GPU only)

    python -m bench.microbench_kernels

None of these ops is expensive. They matter because each one launches several kernels
and makes several passes over the hidden state, twice per decoder layer, and a decode
step on a 1B model is only a few milliseconds of GPU work behind a pile of launches.
The honest way to present them is per-step cost, which is what the last column does.
"""

from __future__ import annotations

import argparse
import time

import torch

from paged_inference_engine.kernels import cache_ops, fused


def timeit(fn, warmup=20, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tokens", type=int, nargs="+", default=[1, 8, 64, 512, 8192])
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--intermediate", type=int, default=8192)
    ap.add_argument("--num-heads", type=int, default=32)
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=16)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    dev, dt = "cuda", torch.bfloat16

    hdr = (f"{'op':>20} {'tokens':>8} {'torch us':>10} {'triton us':>10} "
           f"{'speedup':>8} {'us/step saved':>14}")
    print(hdr)
    print("-" * len(hdr))
    per_step = {"rms_norm": 2, "silu_and_mul": 1, "rope": 1, "reshape_and_cache": 1}

    for n in args.num_tokens:
        x = torch.randn(n, args.hidden, dtype=dt, device=dev)
        res = torch.randn_like(x)
        w = torch.randn(args.hidden, dtype=dt, device=dev)
        gate_up = torch.randn(n, 2 * args.intermediate, dtype=dt, device=dev)
        q = torch.randn(n, args.num_heads, args.head_dim, dtype=dt, device=dev)
        k = torch.randn(n, args.num_kv_heads, args.head_dim, dtype=dt, device=dev)
        v = torch.randn_like(k)
        cos = torch.randn(n, args.head_dim // 2, dtype=dt, device=dev)
        sin = torch.randn_like(cos)
        nb = max(4, (n + args.block_size - 1) // args.block_size + 2)
        kc = torch.zeros(nb, args.block_size, args.num_kv_heads, args.head_dim,
                         dtype=dt, device=dev)
        vc = torch.zeros_like(kc)
        slots = torch.randperm(nb * args.block_size, device=dev)[:n].to(torch.int64)

        cases = [
            ("rms_norm", lambda: fused.fused_add_rms_norm_torch(x, res.clone(), w, 1e-5),
             lambda: fused.fused_add_rms_norm(x, res.clone(), w, 1e-5)),
            ("silu_and_mul", lambda: fused.silu_and_mul_torch(gate_up),
             lambda: fused.silu_and_mul(gate_up)),
            ("rope", lambda: fused.apply_rope_torch(q, k, cos, sin),
             lambda: fused.apply_rope(q.clone(), k.clone(), cos, sin)),
