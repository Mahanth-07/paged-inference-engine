"""Scheduler tests. Also laptop-only. These encode the properties you must not break
while tuning policy.
"""

from paged_inference_engine.block_manager import BlockManager
from paged_inference_engine.config import SchedulerConfig
from paged_inference_engine.sampling import SamplingParams
from paged_inference_engine.scheduler import Scheduler
from paged_inference_engine.sequence import Sequence

BLOCK_SIZE = 16


def make_seq(prompt_len, max_tokens=8):
    return Sequence(
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
    )


def make_scheduler(num_blocks=256, **cfg_kwargs):
    cfg = SchedulerConfig(**cfg_kwargs)
    bm = BlockManager(num_blocks=num_blocks, block_size=BLOCK_SIZE)
    return Scheduler(cfg, bm), bm


def run_step(sched, bm):
    """Apply a scheduler output the way the engine would."""
    out = sched.schedule()
    for s in out.prefills:
        s.seq.num_computed_tokens += s.num_tokens
        if s.seq.num_computed_tokens >= s.seq.prompt_len:
            s.seq.append_token(0)
    for s in out.decodes:
        s.seq.append_token(0)
    finished = sched.free_finished(eos_token_id=None)
    return out, finished


# ---- basic behavior -------------------------------------------------------


def test_respects_token_budget():
    sched, bm = make_scheduler(max_num_batched_tokens=100, enable_chunked_prefill=True)
    for _ in range(10):
        sched.add_seq(make_seq(64))
    out = sched.schedule()
    assert out.num_batched_tokens <= 100


def test_respects_max_num_seqs():
    sched, bm = make_scheduler(max_num_seqs=4, max_num_batched_tokens=100_000)
    for _ in range(20):
        sched.add_seq(make_seq(16))
    sched.schedule()
    assert len(sched.running) <= 4


def test_never_overallocates_blocks():
    sched, bm = make_scheduler(num_blocks=32, max_num_batched_tokens=100_000)
    for _ in range(50):
        sched.add_seq(make_seq(64, max_tokens=32))
    for _ in range(300):
        run_step(sched, bm)
        bm.allocator.check_invariants()
        assert bm.num_free_blocks >= 0


def test_all_requests_eventually_finish():
    """No starvation and no deadlock -- the two failure modes that will waste a whole
    GPU session if you find them there instead of here."""
    sched, bm = make_scheduler(num_blocks=64, max_num_batched_tokens=512, max_num_seqs=8)
    seqs = [make_seq(pl, max_tokens=ml) for pl, ml in
            [(10, 5), (200, 20), (33, 40), (17, 3), (300, 10), (8, 60), (128, 12)]]
    for s in seqs:
        sched.add_seq(s)

    finished, steps = [], 0
    while sched.has_unfinished and steps < 20_000:
        steps += 1
        _, done = run_step(sched, bm)
        finished.extend(done)

    assert len(finished) == len(seqs), f"only {len(finished)}/{len(seqs)} finished in {steps} steps"
