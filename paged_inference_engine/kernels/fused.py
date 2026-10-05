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
