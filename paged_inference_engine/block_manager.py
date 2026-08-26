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
