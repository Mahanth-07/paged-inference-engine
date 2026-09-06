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
