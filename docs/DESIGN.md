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

### Sizing

```
bytes_per_block = 2 (K and V) × block_size × num_kv_heads × head_dim × dtype_bytes × num_layers
num_blocks      = (total_vram × utilization − weights − activation_peak) / bytes_per_block
```

`activation_peak` is **measured**, not guessed: `ModelRunner.profile_num_blocks()` runs
one forward pass at exactly `max_num_batched_tokens` and reads
`torch.cuda.max_memory_allocated()`. Guessing here is the classic source of "works at
batch 8, OOMs at batch 64", because the peak scales with the batch width and a constant
guess is only right at one width.

---

## 2. Block manager

### The three states of a block

Most descriptions of paged attention give a block two states. It needs three, and the
third is what makes prefix caching work across requests rather than only within one.

| state | meaning |
|---|---|
| **allocated** | refcount ≥ 1; somebody is reading or writing it |
| **free** | nobody wants it; contents are garbage |
| **evictable** | refcount hit 0, but the block still holds valid content under a known hash |

An evictable block is available — `allocate()` will take it when the free list is empty,
LRU-first — but until the memory is actually needed, a later request with the same
prefix can reclaim it for free. Without this state the cache is emptied every time a
request finishes, which on a chat workload is exactly when the next request with the
same system prompt arrives.

### Copy-on-write

`append_slot()` returns a `(src, dst)` pair when the sequence's last block is shared and
is about to be written. The engine performs the copy across every layer's cache
**before** the forward pass, because the forward pass is what writes the new K/V. Doing
it after would corrupt whoever else is still reading that block.

### Prefix caching, and the two rules that make it correct

1. **Hashes are chained.** `hash(block_i) = hash(hash(block_{i-1}), tokens_i)`. Without
   chaining, the same 16 tokens hash identically whether they are the first block of a
   document or the fifth — but KV depends on the entire preceding context, so two
   sequences with matching block content and different prefixes would share KV and the
   output would be silently wrong. The chain makes the hash identify the whole *prefix*
   ending at that block, which is the thing that actually determines the KV.

2. **Publish only after computing.** A block's hash is registered *after* the forward
   pass that filled it, never at allocation time. Registering early lets a concurrent
   request hit the hash and read KV that has not been written yet. Both rules have a
   dedicated test; the second one is invisible until you have two requests in flight.

Reusing a cached block also *requires* skipping its prefill — `num_computed_tokens` is
advanced past the reused prefix. That is not only the throughput win; it is a
correctness requirement, because a reused block may be shared with a live sequence and
recomputing into it would corrupt that reader.

Finally, the cacheable prefix is capped at `(prompt_len - 1) // block_size` blocks. A
sequence whose *entire* prompt came from the cache would have nothing to compute, go
straight to the decode queue, and never produce a first token.

---

## 3. Scheduler

`Scheduler.schedule()` runs once per forward pass and returns which sequences prefill,
which decode, and which get preempted, under three simultaneous budgets:
`max_num_batched_tokens`, `max_num_seqs`, and free blocks.

### Order of operations, and why

1. **Decodes first.** Running sequences have already paid their TTFT; starving them
   spikes ITL, which users feel far more than a slightly worse TTFT. (Flip this with
   the simulator and watch the TTFT/ITL trade move — it is a five-second experiment.)
2. **Preempt from the tail** of the running list when the cache is full. Preempting the
   *oldest* sequence would throw away the most computed work.
3. **Resume partially-prefilled sequences** before admitting new ones, so a chunked
   prompt cannot be starved by a stream of new arrivals.
4. **Admit from the waiting queue**, strictly FCFS. Skipping ahead past a request that
   does not fit is how you build a starvation bug that only appears under load.

### Continuous vs static batching

Static batching (what HF `generate` does) picks N prompts and runs until the *longest*
one finishes. Every sequence that finished early holds its slot for the rest of the
batch. `bench/baselines/hf_generate.py` measures this directly:

```
batch_utilization = sum(output_lens) / (batch_size × max(output_lens))
```

On ShareGPT that is routinely under 40%. That single number is the entire argument for
continuous batching, and it is far more persuasive than a speedup ratio because it
explains *where* the speedup comes from.

Continuous batching reschedules every forward pass: a sequence that hits EOS is evicted
immediately and a waiting request takes its slot on the very next step.

