"""Sampler tests. All CPU.

The sampler is easy to get subtly wrong in ways that never crash: an off-by-one in the
top-p cutoff, top-k that keeps k+1 tokens, a greedy row that goes through the
temperature divide and produces inf. These check the shape of the surviving
distribution, not just that a token came out.
"""

import pytest

