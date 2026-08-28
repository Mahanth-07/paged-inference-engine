"""Paged KV cache memory management.

Pure Python with no torch dependency, so allocation can be tested without a GPU.

Mental model
------------
Physical memory is a flat pool of `num_blocks` blocks, each holding `block_size`
tokens' worth of K and V for every layer. A sequence owns a *block table*: a list
mapping its logical block i to some physical block id. Blocks need not be contiguous,
which is the entire point -- it removes internal fragmentation and lets sequences
grow without reserving max_model_len up front.

Reference counting exists so two sequences can share blocks: identical prefixes
(prefix caching) or forked sequences (beam search / parallel sampling). A block
returns to the free list only when its refcount hits zero.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Iterable, Optional

from .sequence import Sequence


class OutOfBlocksError(RuntimeError):
    pass


class BlockAllocator:
    """Free-list allocator over physical block ids with reference counting.

    Three states a block can be in, and the third one is what makes prefix caching
    work across requests:

      allocated  refcount >= 1, someone is reading or writing it
      free       nobody wants it, contents are garbage
      evictable  refcount hit 0 but the block still holds valid cached content under a
                 known hash. It is available -- allocation will take it -- but until
                 something actually needs the memory, a later request with the same
                 prefix can reclaim it for free.

    Without the evictable state, the cache is emptied every time a request finishes,
    which is exactly when a chat workload's next request (same system prompt) shows up.
    """

    def __init__(self, num_blocks: int):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = num_blocks
        self._free: deque[int] = deque(range(num_blocks))
        self._ref_count: dict[int, int] = {}
        self._evictable: OrderedDict[int, int] = OrderedDict()   # block_id -> content hash
        self._block_hash: dict[int, int] = {}                    # block_id -> content hash
        self._hash_to_block: dict[int, int] = {}                 # content hash -> block_id
        self.num_evictions = 0

    # ---- core ops ---------------------------------------------------------

    def allocate(self) -> int:
        """Take a free block; if there are none, evict the least recently used cached one."""
        if self._free:
            block_id = self._free.popleft()
        elif self._evictable:
            block_id, _ = self._evictable.popitem(last=False)     # LRU
            self._drop_hash(block_id)
            self.num_evictions += 1
        else:
            raise OutOfBlocksError("no free blocks")
        self._ref_count[block_id] = 1
        return block_id

    # ---- content hashing (prefix caching) ---------------------------------

    def register_hash(self, block_id: int, content_hash: int) -> None:
        """Publish an allocated block as holding known, ALREADY COMPUTED content.

        Two rules, both load-bearing:
          - only FULL blocks -- a partial block's contents can still change;
          - only AFTER the forward pass that filled it. Publishing at allocation time
            is a real bug: a second request could hit the hash and read KV that has
            not been written yet, and the resulting output is fluent garbage.
        """
        self._block_hash[block_id] = content_hash
        self._hash_to_block[content_hash] = block_id

    def block_hash(self, block_id: int) -> Optional[int]:
        return self._block_hash.get(block_id)

    def lookup_hash(self, content_hash: int) -> Optional[int]:
        """Resolve a content hash to a block that still genuinely holds that content."""
        block_id = self._hash_to_block.get(content_hash)
        if block_id is None:
            return None
        if self._block_hash.get(block_id) != content_hash:
            self._hash_to_block.pop(content_hash, None)   # stale entry, clean it up
            return None
        return block_id

    def _drop_hash(self, block_id: int) -> None:
        h = self._block_hash.pop(block_id, None)
        if h is not None and self._hash_to_block.get(h) == block_id:
            del self._hash_to_block[h]

    def touch(self, block_id: int) -> bool:
        """Reclaim an evictable block by increffing it. Returns False if it is gone."""
        if block_id in self._evictable:
            del self._evictable[block_id]
            self._ref_count[block_id] = 1
            return True
        if block_id in self._ref_count:
            self._ref_count[block_id] += 1
            return True
        return False

    def allocate_many(self, n: int) -> list[int]:
        if n > len(self._free):
            raise OutOfBlocksError(f"requested {n} blocks, {len(self._free)} free")
        return [self.allocate() for _ in range(n)]

    def incref(self, block_id: int) -> None:
        if block_id not in self._ref_count:
            raise KeyError(f"block {block_id} is not allocated")
        self._ref_count[block_id] += 1

    def free(self, block_id: int) -> None:
        """Decrement refcount. At zero the block becomes free, or -- if it holds cached
        content -- evictable, which keeps it reusable until the memory is actually needed."""
        rc = self._ref_count.get(block_id)
        if rc is None:
            raise KeyError(f"double free or unallocated block {block_id}")
        if rc == 1:
            del self._ref_count[block_id]
            h = self._block_hash.get(block_id)
            if h is None:
                self._free.append(block_id)
            else:
                self._evictable[block_id] = h
        else:
            self._ref_count[block_id] = rc - 1

    def free_many(self, block_ids: Iterable[int]) -> None:
        for b in block_ids:
            self.free(b)

    # ---- introspection ----------------------------------------------------

    @property
    def num_free_blocks(self) -> int:
        """Blocks the allocator can hand out right now. Evictable blocks count: they are
        reusable memory, they just happen to still hold something useful."""
        return len(self._free) + len(self._evictable)

    @property
    def num_allocated_blocks(self) -> int:
        return len(self._ref_count)

    @property
    def num_evictable_blocks(self) -> int:
        return len(self._evictable)

    def ref_count(self, block_id: int) -> int:
        return self._ref_count.get(block_id, 0)

    def is_shared(self, block_id: int) -> bool:
        return self.ref_count(block_id) > 1

    def check_invariants(self) -> None:
        """Call this from tests after every operation. Cheap insurance."""
        total = len(self._free) + len(self._evictable) + len(self._ref_count)
        assert total == self.num_blocks, (
            f"block leak: {len(self._free)} free + {len(self._evictable)} evictable + "
            f"{len(self._ref_count)} allocated != {self.num_blocks} total"
        )
        assert len(set(self._free)) == len(self._free), "duplicate block in free list"
        assert not (set(self._free) & set(self._ref_count)), "block both free and allocated"
        assert not (set(self._evictable) & set(self._ref_count)), "block both evictable and live"
        assert not (set(self._evictable) & set(self._free)), "block both evictable and free"
        assert all(rc >= 1 for rc in self._ref_count.values()), "non-positive refcount"
        assert all(b in self._block_hash for b in self._evictable), "evictable block lost its hash"


@dataclass
class AllocStatus:
    OK = "ok"
    LATER = "later"       # not enough blocks now, but the request could fit eventually
    NEVER = "never"       # request can never fit; reject it


class BlockManager:
    """Maps sequences to physical blocks. The scheduler asks this before admitting work."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        watermark: float = 0.01,
        enable_prefix_caching: bool = False,
    ):
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)
        self.watermark_blocks = int(watermark * num_blocks)
        """Keep a small reserve free so decodes of already-running sequences can always
        append a slot. Without this, admitting one more prefill can deadlock the batch."""
        self.enable_prefix_caching = enable_prefix_caching
        self.num_cache_hit_blocks = 0
        self.num_cache_query_blocks = 0

    # ---- admission --------------------------------------------------------

    def _blocks_for(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, seq: Sequence) -> str:
        # +1: admitting a sequence commits us to writing at least one generated token
        # into the cache. Budgeting only for the prompt is how you deadlock at high load.
        need = self._blocks_for(seq.total_len + 1)
        if need > self.allocator.num_blocks - self.watermark_blocks:
            return AllocStatus.NEVER
        if need <= self.allocator.num_free_blocks - self.watermark_blocks:
            return AllocStatus.OK
        return AllocStatus.LATER

    def allocate(self, seq: Sequence) -> None:
        """Allocate the full block table for a sequence's current token count."""
        assert not seq.block_table, "sequence already has blocks"
        need = seq.num_blocks(self.block_size)
        if self.enable_prefix_caching:
            seq.block_table = self._allocate_with_prefix_cache(seq, need)
        else:
            seq.block_table = self.allocator.allocate_many(need)

    def can_append_slot(self, seq: Sequence) -> bool:
        """Is there room to write ONE more token's KV for this sequence?

        Call this BEFORE appending the token. Two ways it can need a free block:
        crossing a block boundary, or copy-on-write of a shared final block.
        """
        needed = self._blocks_for(seq.total_len + 1)
        if needed > len(seq.block_table):
            return self.allocator.num_free_blocks > 0
        if seq.block_table and self.allocator.is_shared(seq.block_table[-1]):
            return self.allocator.num_free_blocks > 0
        return True

    def append_slot(self, seq: Sequence) -> Optional[tuple[int, int]]:
        """Reserve room for one more token. Returns (src, dst) if a copy-on-write happened.

        CoW case: the sequence's last block is shared (refcount > 1) and we are about to
        write into it. Copy it first or we corrupt whoever else is reading it.
        """
        needed = self._blocks_for(seq.total_len + 1)
        if needed > len(seq.block_table):
            seq.block_table.append(self.allocator.allocate())
            return None

        if not seq.block_table:
            return None
        last = seq.block_table[-1]
        if self.allocator.is_shared(last):
            new_block = self.allocator.allocate()
            seq.block_table[-1] = new_block
            self.allocator.free(last)
            return (last, new_block)
        return None

    def fork(self, parent: Sequence, child: Sequence) -> None:
        """Share the parent's blocks with the child; CoW handles divergence later."""
        child.block_table = list(parent.block_table)
        for block_id in child.block_table:
            self.allocator.incref(block_id)

    def free(self, seq: Sequence) -> None:
        if not seq.block_table:
            return
        self.allocator.free_many(seq.block_table)
        seq.block_table = []

    # ---- slot mapping -----------------------------------------------------

    def slot_mapping(self, seq: Sequence, start: int, end: int) -> list[int]:
        """Flat cache indices for logical token positions [start, end).

        Feed this to reshape_and_cache so the model writes each new K/V into the
        right physical location. Off-by-one bugs here produce output that is
        *almost* right, which is the worst kind of bug -- test it directly.
        """
        return [seq.slot_for_token(i, self.block_size) for i in range(start, end)]

    # ---- FlashInfer-style page table --------------------------------------

    def paged_kv_metadata(self, seqs: list[Sequence]) -> dict[str, list[int]]:
        """Build the CSR page table FlashInfer expects.

        kv_indptr:        (num_seqs + 1,) cumulative page counts
        kv_indices:       concatenated physical block ids
        kv_last_page_len: (num_seqs,) tokens used in each sequence's final page, in [1, page_size]

        All three must be int32 on the GPU. Build them once per step; do not rebuild
        per layer.
        """
        indptr = [0]
        indices: list[int] = []
        last_page_len: list[int] = []
        for seq in seqs:
            indices.extend(seq.block_table)
            indptr.append(len(indices))
            last_page_len.append(seq.last_block_occupancy(self.block_size))
        return {
            "kv_indptr": indptr,
            "kv_indices": indices,
            "kv_last_page_len": last_page_len,
        }

    def block_tables_padded(self, seqs: list[Sequence]) -> list[list[int]]:
        """Dense (num_seqs, max_num_blocks) block table, padded with 0.

        This is the layout vLLM's paged attention kernel wants. Your Triton kernel
        can take either; dense is simpler to index, CSR wastes less memory.
        """
        max_blocks = max((len(s.block_table) for s in seqs), default=0)
        return [s.block_table + [0] * (max_blocks - len(s.block_table)) for s in seqs]

    # ---- prefix caching ---------------------------------------------------

    def block_content_hash(self, tokens: list[int], prev_hash: Optional[int]) -> int:
        """Hash a full block's contents, CHAINED with the previous block's hash.

        The chaining is not optional. Without it, block ["the", "cat", ...] hashes the
        same whether it is the first block of a document or the fifth, so two sequences
        with the same block content but different prefixes would share KV -- and KV
        depends on the entire preceding context, so the output would be silently wrong.
        Chaining makes the hash identify the whole prefix ending at this block.
        """
        return hash((prev_hash, tuple(tokens)))

    def _num_cacheable_blocks(self, seq: Sequence) -> int:
        """How many of a sequence's leading blocks may be served from the cache.

        Only FULL blocks of the PROMPT, and never so many that nothing is left to
        compute: a sequence whose entire prompt came from the cache would go straight
        to the decode queue without ever running a forward pass, and would therefore
        never produce a first token. The `- 1` keeps at least one token of real work.
        """
        return max(0, (seq.prompt_len - 1) // self.block_size)

    def _allocate_with_prefix_cache(self, seq: Sequence, need: int) -> list[int]:
        """Reuse physical blocks whose content hash matches; incref instead of copy.

        Sets seq.num_computed_tokens to the length of the reused prefix, which is what
        actually buys the throughput -- those tokens are never prefilled again. It is
        also a CORRECTNESS requirement, not just an optimisation: a reused block may be
        shared with a live sequence, and recomputing into it would corrupt that reader.
        """
        tokens = seq.all_token_ids
        table: list[int] = []
        prev_hash: Optional[int] = None
        cacheable = self._num_cacheable_blocks(seq)
        matching = True   # the first miss breaks the chain; nothing after it can hit
        num_hit_blocks = 0

        for i in range(need):
            if i >= cacheable:
                table.append(self.allocator.allocate())
                continue
            block_tokens = tokens[i * self.block_size:(i + 1) * self.block_size]
            content_hash = self.block_content_hash(block_tokens, prev_hash)
            self.num_cache_query_blocks += 1
            prev_hash = content_hash

            hit = None
            if matching:
                candidate = self.allocator.lookup_hash(content_hash)
                if candidate is not None and self.allocator.touch(candidate):
                    hit = candidate
            if hit is not None:
                self.num_cache_hit_blocks += 1
                num_hit_blocks += 1
                table.append(hit)
            else:
                matching = False
                table.append(self.allocator.allocate())

        seq.num_computed_tokens = num_hit_blocks * self.block_size
        return table

    def mark_computed_blocks(self, seq: Sequence) -> None:
        """Publish hashes for prompt blocks this step actually finished computing.

        Called by the engine AFTER the forward pass. Registering earlier would let a
        concurrent request read uncomputed KV -- see BlockAllocator.register_hash.
        """
        if not self.enable_prefix_caching:
            return
        tokens = seq.all_token_ids
        prev_hash: Optional[int] = None
        limit = min(self._num_cacheable_blocks(seq), seq.num_computed_tokens // self.block_size)
        for i in range(limit):
            block_tokens = tokens[i * self.block_size:(i + 1) * self.block_size]
            content_hash = self.block_content_hash(block_tokens, prev_hash)
            prev_hash = content_hash
            block_id = seq.block_table[i]
            if self.allocator.block_hash(block_id) is None:
                self.allocator.register_hash(block_id, content_hash)

    @property
    def prefix_cache_hit_rate(self) -> float:
        if not self.num_cache_query_blocks:
            return 0.0
        return self.num_cache_hit_blocks / self.num_cache_query_blocks

    # ---- introspection ----------------------------------------------------

    @property
    def num_free_blocks(self) -> int:
        return self.allocator.num_free_blocks

    @property
    def utilization(self) -> float:
        return self.allocator.num_allocated_blocks / self.allocator.num_blocks


def compute_num_gpu_blocks(
    total_memory_bytes: int,
    weight_bytes: int,
    activation_peak_bytes: int,
    num_layers: int,
