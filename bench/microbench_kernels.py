"""Fused-kernel microbenchmarks: is the Triton version actually worth it? (GPU only)

    python -m bench.microbench_kernels

None of these ops is expensive. They matter because each one launches several kernels
and makes several passes over the hidden state, twice per decoder layer, and a decode
step on a 1B model is only a few milliseconds of GPU work behind a pile of launches.
The honest way to present them is per-step cost, which is what the last column does.
"""

from __future__ import annotations

import argparse
