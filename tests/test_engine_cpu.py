"""End-to-end engine correctness on the CPU fixture model.

These tests need no GPU and check block tables, slot
mapping, chunked prefill, preemption, cross-sequence contamination, and prefix cache
sharing. The oracle is `ReferenceGenerator`, which has no cache at all and therefore
cannot share a bug with the paged path.

Everything here is greedy, so "correct" means token-for-token identical, not close.
"""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from paged_inference_engine.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig  # noqa: E402
from paged_inference_engine.engine import LLMEngine  # noqa: E402
from paged_inference_engine.models.reference import ReferenceGenerator  # noqa: E402
from paged_inference_engine.sampling import SamplingParams  # noqa: E402

GREEDY = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True)


def make_engine(**overrides) -> LLMEngine:
    cfg = EngineConfig(
        model=ModelConfig(model="__random__", dtype="float32", max_model_len=512,
                          hidden_size=128, num_layers=2, num_heads=4, num_kv_heads=2,
                          head_dim=32, intermediate_size=256, vocab_size=1000, seed=0),
        cache=CacheConfig(block_size=overrides.pop("block_size", 16),
                          num_gpu_blocks=overrides.pop("num_gpu_blocks", 256),
                          enable_prefix_caching=overrides.pop("prefix_caching", False)),
        scheduler=SchedulerConfig(
            max_num_batched_tokens=overrides.pop("max_num_batched_tokens", 128),
            max_num_seqs=overrides.pop("max_num_seqs", 8),
            enable_chunked_prefill=overrides.pop("chunked_prefill", True)),
        attention_backend="naive", device="cpu",
    )
    return LLMEngine(cfg)


PROMPTS = [
    [5, 6, 7, 8, 9, 10, 11, 12, 13],
    [1, 2, 3],
    list(range(20, 60)),
    [7] * 35,
    list(range(200, 217)),          # 17 tokens: one past a 16-block boundary
    list(range(300, 332)),          # 32 tokens: exactly two blocks
]


def gen_outputs(engine, prompts, sp=GREEDY):
    return engine.generate(prompt_token_ids=copy.deepcopy(prompts), sampling_params=sp)


def gen(engine, prompts, sp=GREEDY):
    return [o.token_ids for o in gen_outputs(engine, prompts, sp)]


# ---------------------------------------------------------------------------


def test_paged_engine_matches_cacheless_reference():
    """The M1 oracle. If this fails, nothing below it means anything."""
    engine = make_engine()
    ref = ReferenceGenerator(engine.model_runner.model, "cpu")
    got = gen(engine, PROMPTS)
    for prompt, tokens in zip(PROMPTS, got):
        assert tokens == ref.greedy(prompt, GREEDY.max_tokens), f"diverged on {prompt[:5]}..."


def test_logits_match_reference_not_just_argmax():
    """argmax agreement is a weak signal on a random-weight model: a broken attention
    can still pick the same token. Compare the logits themselves."""
    engine = make_engine()
    ref = ReferenceGenerator(engine.model_runner.model, "cpu")
    prompt = list(range(20, 60))
    want = ref.forward(prompt)[-1]

    runner = engine.model_runner
    engine.add_request(prompt_token_ids=prompt, sampling_params=GREEDY)
    sched = engine.scheduler.schedule()
    mi, _ = runner.prepare_inputs(sched, engine.block_manager)
    with torch.inference_mode():
        hidden = runner.model(mi.input_ids, mi.positions, runner.kv_caches, mi)
        got = runner.model.compute_logits(hidden, mi.logits_indices)[0]
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_batched_matches_unbatched():
    """The nastiest class of bug in the project: cross-sequence contamination from a
    wrong block table, attention mask, or query_start_loc. The output stays fluent, so
    reading it tells you nothing."""
    solo = []
    for p in PROMPTS:
        e = make_engine(max_num_seqs=1, max_num_batched_tokens=512)
        solo.append(gen(e, [p])[0])
    batched = gen(make_engine(max_num_seqs=8, max_num_batched_tokens=64), PROMPTS)
    assert batched == solo


def test_many_sequences_of_mixed_lengths():
    """32 concurrent sequences with lengths that straddle block boundaries."""
    prompts = [list(range(100, 100 + n)) for n in (1, 15, 16, 17, 31, 32, 33, 64)] * 4
    solo = [gen(make_engine(max_num_seqs=1, max_num_batched_tokens=512), [p])[0]
            for p in prompts[:8]]
    batched = gen(make_engine(max_num_seqs=32, max_num_batched_tokens=256), prompts)
    assert batched[:8] == solo
    assert batched[:8] == batched[8:16] == batched[16:24] == batched[24:32]


@pytest.mark.parametrize("block_size", [8, 16, 32])
def test_block_size_does_not_change_output(block_size):
    """Block size is a memory/efficiency knob. If it changes the answer, the slot
    mapping or the last-page masking is wrong."""
    baseline = gen(make_engine(block_size=16), PROMPTS)
    assert gen(make_engine(block_size=block_size), PROMPTS) == baseline


@pytest.mark.parametrize("chunked", [True, False])
def test_chunked_prefill_does_not_change_output(chunked):
    """Chunked prefill splits a prompt across steps, so the second chunk's queries sit
    at absolute positions that are not 0..n. Getting that wrong is invisible unless you
    diff against the unchunked run."""
    baseline = gen(make_engine(chunked_prefill=True, max_num_batched_tokens=512), PROMPTS)
    got = gen(make_engine(chunked_prefill=chunked, max_num_batched_tokens=16), PROMPTS) \
        if chunked else gen(make_engine(chunked_prefill=False,
                                        max_num_batched_tokens=512), PROMPTS)
    assert got == baseline


@pytest.mark.parametrize("budget", [1, 2, 3, 5])
def test_tiny_chunks_still_correct(budget):
    """Prefill a couple of tokens at a time. Maximally adversarial for the absolute-
    position bookkeeping and for the bottom-right causal alignment."""
    baseline = gen(make_engine(max_num_batched_tokens=512), PROMPTS)
    assert gen(make_engine(max_num_batched_tokens=budget, max_num_seqs=4), PROMPTS) == baseline


# Long outputs against a tight cache: the pressure has to arrive DURING decode, because
# admission control already refuses to over-commit at prefill time.
LONG = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True)


def test_preemption_is_output_preserving():
    """Run with a KV cache small enough to force preemption; the output must be
    identical to an unpreempted run. If it is not, the recompute path is losing or
    duplicating tokens."""
    baseline = gen(make_engine(num_gpu_blocks=512), PROMPTS, LONG)
    engine = make_engine(num_gpu_blocks=24, max_num_seqs=8, max_num_batched_tokens=64)
    got = gen(engine, PROMPTS, LONG)
    assert engine.scheduler.num_preemptions > 0, "cache was not small enough to preempt"
    assert got == baseline


def test_preemption_does_not_leak_blocks():
    engine = make_engine(num_gpu_blocks=24, max_num_seqs=8, max_num_batched_tokens=64)
    gen(engine, PROMPTS, LONG)
    assert engine.scheduler.num_preemptions > 0
    engine.block_manager.allocator.check_invariants()
    assert engine.block_manager.num_free_blocks == 24


def test_blocks_are_returned_after_every_run():
    engine = make_engine()
    before = engine.block_manager.num_free_blocks
    for _ in range(3):
        gen(engine, PROMPTS)
        engine.block_manager.allocator.check_invariants()
        assert engine.block_manager.num_free_blocks == before


def test_prefix_caching_is_output_preserving_and_hits():
    """Two properties, and the second one is the one people forget: a correct cache
    that never hits passes the equality assertion and buys nothing."""
    system = list(range(100, 180))                      # shared 80-token system prompt
    prompts = [system + [i, i + 1, i + 2] for i in range(5, 12)]

    off = gen(make_engine(num_gpu_blocks=512), prompts)
    engine = make_engine(num_gpu_blocks=512, prefix_caching=True)
    on = gen(engine, prompts)
    assert on == off
    hit_rate = engine.stats()["prefix_cache_hit_rate"]
    assert hit_rate > 0.5, f"prefix cache barely hit: {hit_rate:.2f}"


def test_prefix_cache_does_not_confuse_identical_blocks_in_different_contexts():
    """Block hashes are CHAINED with the previous block's hash. Without chaining, the
    same 16 tokens appearing after a different prefix would wrongly share KV -- and KV
    depends on the whole preceding context, so the output would be silently wrong."""
    common = list(range(400, 416))                      # one full block, appears twice
    a = list(range(1, 17)) + common + [50, 51]
    b = list(range(30, 46)) + common + [50, 51]
    off = gen(make_engine(num_gpu_blocks=512), [a, b])
    on = gen(make_engine(num_gpu_blocks=512, prefix_caching=True), [a, b])
    assert on == off
    assert off[0] != off[1], "fixture is degenerate: the two prefixes gave the same answer"


def test_finished_sequence_leaves_the_batch_immediately():
    """Iteration-level batching: a request that finishes must free its slot on the very
    next step, not when the longest member of some batch finishes. That difference is
    the entire throughput argument against HF `generate`."""
    engine = make_engine(max_num_seqs=4, max_num_batched_tokens=256)
    short = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True)
    long = SamplingParams(temperature=0.0, max_tokens=30, ignore_eos=True)
    engine.add_request(prompt_token_ids=[1, 2, 3], sampling_params=short)
    engine.add_request(prompt_token_ids=[4, 5, 6], sampling_params=long)

    running_after_short_finished = None
    for _ in range(40):
        outs = engine.step()
        if any(o.finished for o in outs) and running_after_short_finished is None:
            running_after_short_finished = len(engine.scheduler.running)
        if not engine.has_unfinished_requests():
            break
    assert running_after_short_finished == 1, "finished sequence kept its slot"


def test_waiting_request_joins_on_the_next_step():
    engine = make_engine(max_num_seqs=1, max_num_batched_tokens=256)
    short = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True)
    engine.add_request(prompt_token_ids=[1, 2, 3], sampling_params=short)
    engine.add_request(prompt_token_ids=[4, 5, 6], sampling_params=short)
    seen_second = False
    for _ in range(20):
        engine.step()
        if len(engine.scheduler.waiting) == 0 and engine.scheduler.running:
            seen_second = True
        if not engine.has_unfinished_requests():
            break
    assert seen_second and not engine.has_unfinished_requests()


def test_streaming_emits_one_output_per_token():
    engine = make_engine()
    engine.add_request(prompt_token_ids=[5, 6, 7], sampling_params=GREEDY)
    emitted = 0
    while engine.has_unfinished_requests():
        for out in engine.step():
            emitted += len(out.new_token_ids) if not out.finished or out.new_token_ids else 0
    assert emitted == GREEDY.max_tokens


def test_max_tokens_is_clamped_to_max_model_len():
    engine = make_engine()
    engine.config.model.max_model_len = 64
    engine.max_model_len = 64
    sid = engine.add_request(prompt_token_ids=list(range(60)),
                             sampling_params=SamplingParams(max_tokens=1000,
                                                            ignore_eos=True))
    seq = next(s for s in engine.scheduler.waiting if s.seq_id == sid)
    assert seq.sampling_params.max_tokens == 4

    with pytest.raises(ValueError):
        engine.add_request(prompt_token_ids=list(range(70)),
                           sampling_params=SamplingParams())


# ---------------------------------------------------------------------------
# admission control: unschedulable requests must fail loudly, not stall
# ---------------------------------------------------------------------------


def test_oversized_prompt_is_rejected_not_stalled_without_chunked_prefill():
    """Without chunked prefill, a prompt wider than the whole token budget can never be
    scheduled. Under FCFS it would block every request behind it forever. It must be
