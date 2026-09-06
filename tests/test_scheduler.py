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
