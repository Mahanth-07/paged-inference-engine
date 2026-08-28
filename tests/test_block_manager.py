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


@pytest.mark.parametrize("total_len,expected", [(1, 1), (15, 15), (16, 16), (17, 1), (32, 16), (33, 1)])
def test_last_page_len_is_never_zero(total_len, expected):
    """FlashInfer requires kv_last_page_len in [1, page_size]. A 0 here is a
    silent-wrong-answer bug, not a crash. Guard it with a test."""
    seq = make_seq(total_len)
    assert seq.last_block_occupancy(BLOCK_SIZE) == expected
    assert 1 <= seq.last_block_occupancy(BLOCK_SIZE) <= BLOCK_SIZE


def test_slot_mapping_is_contiguous_within_a_block():
    bm = BlockManager(num_blocks=64, block_size=BLOCK_SIZE)
    seq = make_seq(40)
    bm.allocate(seq)
    slots = bm.slot_mapping(seq, 0, 40)
    assert len(slots) == 40
    assert len(set(slots)) == 40, "slot mapping must be injective or you overwrite KV"
    for i in range(40):
        blk, off = seq.block_table[i // BLOCK_SIZE], i % BLOCK_SIZE
        assert slots[i] == blk * BLOCK_SIZE + off


def test_append_slot_allocates_only_on_block_boundary():
    """Call order is append_slot() THEN append_token() -- you reserve the cache slot
    before the model writes K/V into it."""
    bm = BlockManager(num_blocks=64, block_size=BLOCK_SIZE)
    seq = make_seq(BLOCK_SIZE)  # exactly one full block
    bm.allocate(seq)
    assert len(seq.block_table) == 1
    free_before = bm.num_free_blocks

    bm.append_slot(seq)             # room for token 17 -> needs a 2nd block
    seq.output_token_ids.append(0)
    assert len(seq.block_table) == 2
    assert bm.num_free_blocks == free_before - 1

    bm.append_slot(seq)             # token 18 fits in the block we already have
    seq.output_token_ids.append(0)
    assert len(seq.block_table) == 2
    assert bm.num_free_blocks == free_before - 1


def test_every_written_token_has_a_slot():
    """The invariant that ties the block manager to the kernel: after append_slot,
    slot_for_token must be valid for the token about to be written."""
    bm = BlockManager(num_blocks=64, block_size=BLOCK_SIZE)
    seq = make_seq(30)
    bm.allocate(seq)
    for _ in range(40):
        bm.append_slot(seq)
        slot = seq.slot_for_token(seq.total_len, BLOCK_SIZE)  # must not IndexError
        assert 0 <= slot < 64 * BLOCK_SIZE
        seq.output_token_ids.append(0)


def test_fork_shares_blocks_and_cow_on_append():
    bm = BlockManager(num_blocks=64, block_size=BLOCK_SIZE)
    parent = make_seq(BLOCK_SIZE * 2)
    bm.allocate(parent)
    child = make_seq(BLOCK_SIZE * 2)
    bm.fork(parent, child)

