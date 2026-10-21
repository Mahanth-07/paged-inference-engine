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
