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
              SamplingParams(temperature=0.0)]
    out = sample(logits, params)
    assert out[0] == logits[0].argmax()
    assert out[2] == logits[2].argmax()


def test_top_k_restricts_support():
    """Only the k highest-logit tokens may ever be sampled."""
    logits = _logits(1, vocab=16)
    k = 3
    allowed = set(logits[0].topk(k).indices.tolist())
    sampler = Sampler(seed=0)
    got = {int(sampler(logits, [SamplingParams(temperature=1.0, top_k=k)])[0])
           for _ in range(300)}
    assert got <= allowed
    assert len(got) > 1, "top-k collapsed to a single token; the test proves nothing"


def test_top_p_restricts_support_and_keeps_the_crossing_token():
    """Nucleus sampling keeps the token that CROSSES p, not just the ones strictly
    below it. Dropping it is an easy off-by-one that shrinks the nucleus by one."""
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    sampler = Sampler(seed=0)
    got = {int(sampler(logits, [SamplingParams(temperature=1.0, top_p=0.7)])[0])
           for _ in range(400)}
    assert got == {0, 1}, got          # 0.5 then 0.8 crosses 0.7 -> both kept


def test_temperature_widens_the_distribution():
    logits = torch.tensor([[3.0, 1.0, 0.5, 0.0]])
    # One sampler per set of draws: a fresh Sampler re-seeds its generator, so
    # constructing one inside the loop would draw the same token 200 times.
    s_cold, s_hot = Sampler(seed=1), Sampler(seed=1)
    cold = {int(s_cold(logits, [SamplingParams(temperature=0.1)])[0]) for _ in range(200)}
    hot = {int(s_hot(logits, [SamplingParams(temperature=5.0)])[0]) for _ in range(200)}
    assert cold == {0}
    assert len(hot) > 1


def test_batched_rows_are_independent():
    """Row i's parameters must not leak into row j. Build a batch where each row has a
