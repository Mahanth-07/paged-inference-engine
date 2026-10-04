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


