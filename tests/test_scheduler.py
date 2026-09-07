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
    for s in seqs:
        assert s.output_len == s.sampling_params.max_tokens


def test_finished_sequence_frees_its_slot_immediately():
    """This is the definition of continuous batching. If a finished sequence lingers
    in `running`, you have static batching with extra steps."""
    sched, bm = make_scheduler(max_num_seqs=2, max_num_batched_tokens=100_000)
    short, long_ = make_seq(16, max_tokens=1), make_seq(16, max_tokens=50)
    waiting = make_seq(16, max_tokens=5)
    sched.add_seq(short)
    sched.add_seq(long_)
    sched.add_seq(waiting)

    run_step(sched, bm)   # prefill both; short emits its only token
    run_step(sched, bm)   # short is finished and evicted
    assert short not in sched.running
    run_step(sched, bm)
    assert waiting in sched.running, "waiting request should take the freed slot at once"


# ---- chunked prefill ------------------------------------------------------


def test_chunked_prefill_splits_a_long_prompt():
    sched, bm = make_scheduler(max_num_batched_tokens=64, enable_chunked_prefill=True)
    seq = make_seq(256)
    sched.add_seq(seq)
    out = sched.schedule()
    assert len(out.prefills) == 1
    assert out.prefills[0].num_tokens == 64, "prompt should be chunked to the budget"
    assert seq.is_prefill


def test_without_chunked_prefill_an_oversized_prompt_waits():
    sched, bm = make_scheduler(max_num_batched_tokens=64, enable_chunked_prefill=False)
    sched.add_seq(make_seq(256))
    out = sched.schedule()
    assert out.is_empty


def test_chunked_prefill_does_not_stall_decodes():
    """The whole point: a 4000-token prompt arriving must not block running decodes."""
    sched, bm = make_scheduler(num_blocks=2048, max_num_batched_tokens=512,
                               enable_chunked_prefill=True)
    running = [make_seq(16, max_tokens=100) for _ in range(4)]
    for s in running:
        sched.add_seq(s)
    run_step(sched, bm)

    sched.add_seq(make_seq(4000, max_tokens=5))
    for _ in range(5):
        out, _ = run_step(sched, bm)
        assert len(out.decodes) == 4, "decodes must keep running alongside the chunked prefill"


# ---- preemption -----------------------------------------------------------


def test_preemption_under_memory_pressure_is_recoverable():
    sched, bm = make_scheduler(num_blocks=24, max_num_batched_tokens=100_000, max_num_seqs=8)
    seqs = [make_seq(48, max_tokens=40) for _ in range(6)]
    for s in seqs:
        sched.add_seq(s)

    finished, steps = [], 0
    while sched.has_unfinished and steps < 50_000:
        steps += 1
        _, done = run_step(sched, bm)
        finished.extend(done)

    assert len(finished) == len(seqs)
    assert sched.num_preemptions > 0, "this config should be tight enough to force preemption"
    bm.allocator.check_invariants()
    assert bm.num_free_blocks == 24, "all blocks returned after everything finished"


def test_preempted_sequence_keeps_its_generated_tokens():
    from paged_inference_engine.sequence import SequenceStatus

    seq = make_seq(32, max_tokens=10)
    seq.output_token_ids = [1, 2, 3]
    seq.num_computed_tokens = 35
    seq.block_table = [0, 1, 2]
    seq.reset_for_recompute()

    assert seq.output_token_ids == [1, 2, 3], "preemption must not lose generated output"
    assert seq.num_computed_tokens == 0, "but all KV must be recomputed"
    assert seq.block_table == []
    assert seq.status is SequenceStatus.PREEMPTED
    assert seq.num_preemptions == 1
