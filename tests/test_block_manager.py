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


def test_allocate_and_free_roundtrip():
    a = BlockAllocator(8)
    blocks = a.allocate_many(8)
    assert a.num_free_blocks == 0
    with pytest.raises(OutOfBlocksError):
        a.allocate()
    a.free_many(blocks)
    assert a.num_free_blocks == 8
    a.check_invariants()


def test_refcount_defers_free():
    a = BlockAllocator(4)
    b = a.allocate()
    a.incref(b)
    assert a.ref_count(b) == 2 and a.is_shared(b)
    a.free(b)
    assert a.num_free_blocks == 3, "shared block must not return to the free list"
    a.free(b)
    assert a.num_free_blocks == 4
    a.check_invariants()


def test_double_free_raises():
    a = BlockAllocator(2)
    b = a.allocate()
    a.free(b)
    with pytest.raises(KeyError):
        a.free(b)


def test_no_leaks_under_random_churn():
    import random

    rng = random.Random(0)
    a = BlockAllocator(64)
    held = []
    for _ in range(5000):
        if held and (len(held) >= 64 or rng.random() < 0.5):
            a.free(held.pop(rng.randrange(len(held))))
        else:
            try:
                held.append(a.allocate())
            except OutOfBlocksError:
                pass
        a.check_invariants()
    a.free_many(held)
    assert a.num_free_blocks == 64


# ---- block manager --------------------------------------------------------


@pytest.mark.parametrize("prompt_len,expected", [(1, 1), (15, 1), (16, 1), (17, 2), (160, 10), (161, 11)])
def test_block_count_is_ceil(prompt_len, expected):
    bm = BlockManager(num_blocks=64, block_size=BLOCK_SIZE)
    seq = make_seq(prompt_len)
    bm.allocate(seq)
    assert len(seq.block_table) == expected

