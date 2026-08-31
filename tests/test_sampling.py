"""Sampler tests. All CPU.

The sampler is easy to get subtly wrong in ways that never crash: an off-by-one in the
top-p cutoff, top-k that keeps k+1 tokens, a greedy row that goes through the
temperature divide and produces inf. These check the shape of the surviving
distribution, not just that a token came out.
"""

import pytest

torch = pytest.importorskip("torch")

from paged_inference_engine.sampling import Sampler, SamplingParams, sample  # noqa: E402


def _logits(rows, vocab=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, vocab, generator=g)


def test_greedy_is_argmax():
    logits = _logits(4)
    params = [SamplingParams(temperature=0.0)] * 4
    torch.testing.assert_close(sample(logits, params), logits.argmax(-1))


def test_greedy_row_survives_a_mixed_batch():
    """A zero-temperature row in a batch with sampling rows must not go through the
    temperature divide -- that is a divide by zero and produces nan, not argmax."""
    logits = _logits(3)
    params = [SamplingParams(temperature=0.0), SamplingParams(temperature=1.0),
