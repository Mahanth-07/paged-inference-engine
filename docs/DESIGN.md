# Design

Written for someone who knows transformers but not serving. It covers the three things
that make an inference engine an *engine* rather than a `generate()` loop: how KV
memory is managed, how work is scheduled, and how the attention kernel copes with the
first two.

---

## 0. The shape of the system

```
                 ┌──────────────────────────────────────────────┐
   HTTP /v1 ────▶│ APIServer  (FastAPI, SSE streaming)          │
                 └──────────────────┬───────────────────────────┘
                                    │ add_request(prompt, sampling_params)
                 ┌──────────────────▼───────────────────────────┐
                 │ LLMEngine.step()   ← the whole loop is here  │
                 └──────┬───────────────────────────┬───────────┘
                        │                           │
          ┌─────────────▼────────────┐   ┌──────────▼─────────────────┐
          │ Scheduler                │   │ ModelRunner                │
          │  waiting / running queues│   │  build input tensors       │
          │  token + block budget    │   │  block_tables → GPU        │
          │  preemption (recompute)  │   │  forward pass              │
          │  chunked prefill         │   │  CUDA graph replay (decode)│
          └─────────────┬────────────┘   └──────────┬─────────────────┘
                        │ allocate/free              │
          ┌─────────────▼────────────┐   ┌──────────▼─────────────────┐
          │ BlockManager             │   │ Model (Llama)              │
          │  BlockAllocator (free    │   │  RMSNorm, RoPE, SwiGLU     │
          │   list, refcounts, LRU)  │   │  PagedAttention layer      │
          │  BlockTable per sequence │   │   ├ reshape_and_cache      │
          │  prefix-cache hash map   │   │   ├ prefill: paged varlen  │
          └──────────────────────────┘   │   └ decode: paged kernel   │
                                         └────────────────────────────┘
```

**The single most important invariant:** the scheduler never admits work the block
manager cannot back with physical blocks. Every scheduling decision is a memory
decision, and the two components are deliberately coupled through one narrow interface
(`can_allocate` / `can_append_slot`) so that coupling is explicit rather than emergent.

A second invariant, less obvious and the source of the subtlest bug in the project:

> For a sequence in decode, KV exists for positions `[0, total_len - 1)`, and
> `num_computed_tokens == total_len`. The token at `total_len - 1` is the one sampled
> last step; its KV is written by the *next* forward pass.

Everything about chunked prefill, preemption, and slot mapping falls out of that line.

---

## 1. Why paging

Static KV allocation reserves `max_model_len` per sequence. A request that generates 50
tokens against a 4096-token reservation wastes 98.8% of its allocation. Concurrency is
bounded by KV memory, so that waste is a direct throughput ceiling — you do not run out
of compute, you run out of places to put K and V.

Paging allocates 16 tokens at a time:

- **Internal fragmentation** drops to at most one partly-filled block per sequence
  (≤ 15 tokens, vs. thousands).
- **External fragmentation** disappears entirely: all blocks are the same size and
  therefore interchangeable, so there is no such thing as a hole too small to use.
- **Blocks become shareable.** Two sequences can point at the same physical block. That
  is what makes prefix caching (and forking, for beam search) possible at all, and it
  is why the allocator is refcounted rather than a plain free list.

The cost is that K and V for one sequence are no longer contiguous in memory, so the
attention kernel has to gather through a block table. That cost is the subject of §4.
