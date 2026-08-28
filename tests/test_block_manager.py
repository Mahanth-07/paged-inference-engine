"""CPU tests for block allocation and cache bookkeeping.

If these pass and paged output is still wrong, the bug is in the kernel. If these fail,
do not touch the kernel.
"""

import pytest

from paged_inference_engine.block_manager import BlockAllocator, BlockManager, OutOfBlocksError
from paged_inference_engine.sampling import SamplingParams
from paged_inference_engine.sequence import Sequence

BLOCK_SIZE = 16


def make_seq(prompt_len: int, output_len: int = 0) -> Sequence:
    seq = Sequence(
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max(output_len, 1)),
    )
    seq.output_token_ids = list(range(output_len))
    return seq


# ---- allocator ------------------------------------------------------------
