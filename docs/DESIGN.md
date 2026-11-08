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

### Chunked prefill

Without it, one 2000-token prompt monopolises a step and every decoding sequence stalls
behind it — a single ITL spike of tens of milliseconds, right in the p99. With it,
prefill is capped per step and mixed with decodes in the same batch.

The correctness consequence is the one thing to get right: a chunk's queries sit at
absolute positions `ctx - q_len .. ctx - 1`, not `0 .. q_len - 1`. Causal masking must
therefore be **bottom-right aligned**. Top-left alignment (`is_causal=True` on a naive
SDPA call) passes the first chunk and is wrong for every chunk after it, while producing
output that still reads fine.

There is also a failure mode *without* chunked prefill worth knowing: a prompt longer
than `max_num_batched_tokens` can never be scheduled at all, and under strict FCFS it
would block every request behind it forever. So the scheduler rejects it at admission
rather than stalling on it — a clear error beats a silent hang, and the requests behind
it still complete. Measured on a lognormal trace at 12 req/s with a 1024-token budget:

```
     --chunked-prefill:   2084.6 tok/s | completed 150/150 | rejected 0
  --no-chunked-prefill:   1770.6 tok/s | completed 129/150 | rejected 21
```

14% of a realistic trace is simply unservable without chunking. The simulator reports
completions and rejections alongside throughput for exactly this reason: a run that
dropped a seventh of its workload must not be able to look merely "a bit slower".

Rejected requests are drained by the engine every step and turned into a finished
`RequestOutput` with `finish_reason="ignored"`. The alternative — dropping the sequence
silently, which the scaffold's `AllocStatus.NEVER` path did — leaves an HTTP client
hanging until it times out, which is a much worse failure and much harder to debug.

### Preemption

Recompute-based: drop the blocks, keep the generated tokens, re-add to the *front* of
the waiting queue (it has already waited once), and re-prefill on readmission.

Recompute beats swap-to-CPU on modern hardware because prefill is compute-bound and
massively parallel, whereas a swap is a PCIe round trip of the entire context in both
directions — and recompute frees the blocks *immediately*, which is the resource you
were short of.

The bug that lives here, and which this project actually hit: a preempted sequence must
recompute its **prompt and its already-generated tokens**, not just the prompt. Bounding
the re-prefill by `prompt_len` leaves a hole in the cache from the preemption point
onward, and the output stays completely fluent. `is_prefill` is therefore
`num_computed_tokens < total_len`, and `test_preemption_is_output_preserving` exists
specifically to catch it.

---

## 4. Attention

Four backends, all producing identical greedy output (`test_all_backends_agree`):

| backend | prefill | decode | role |
|---|---|---|---|
| `naive` | gather blocks → SDPA | gather blocks → SDPA | the oracle; runs on CPU |
| `flashinfer` | `BatchPrefillWithPagedKVCacheWrapper` | `BatchDecodeWithPagedKVCacheWrapper` | fast, unblocks M3/M4 |
| `flash_attn` | `flash_attn_varlen_func` + block table | `flash_attn_with_kvcache` | second opinion |
| `triton` | our FA2-style paged prefill kernel | our paged decode kernel | the point of the exercise |

The `naive` backend is not scaffolding to be deleted. It differs from the fast kernels
in exactly one way — it does not touch the block table during the math — which is what
makes it a useful oracle when paged output goes subtly wrong.

FlashInfer's wrappers are owned by `AttentionState` on the `ModelRunner`, not by the
attention layers, because `plan()` depends only on the step's page table and not on the
layer. Planning once per step instead of once per layer is a 16× reduction in that cost
on a 1B model.

### The decode kernel

Decode has query length 1. There is no large GEMM; it is a memory sweep over the KV
cache, and arithmetic intensity is roughly the batch size. So it is **memory bound**, and
the optimisation target is percent of peak HBM bandwidth, not TFLOPs.
`bench/microbench_attention.py` reports achieved GB/s against a *measured* device
ceiling, which is the honest way to state it.

Two design decisions carry most of the performance:

**GQA head packing.** One program handles one `(sequence, kv_head)` pair and processes
all `GQA_RATIO` query heads that share that KV head together. They read the same K and V,
so this turns `GQA_RATIO` separate sweeps of the cache into one. On Llama-3.2-1B (ratio
4) that is a 4× cut in KV traffic, and KV traffic is essentially the entire cost.

**The KV tile is decoupled from the page size.** A tile gathers `TILE_KV // block_size`
pages at once:

```
idx      = arange(TILE_KV)
page     = block_table[seq, t * pages_per_tile + idx // block_size]
offset   = idx % block_size
address  = page * stride_page + offset * stride_token + kv_head * stride_head
```

This is a fully general gather, which buys three things: `block_size = 8` works at all
(a 8-wide `tl.dot` is illegal), 16-element loads become 64-element ones that actually
coalesce, and the page size becomes a pure memory-policy knob with no kernel
consequences — which is what makes the block-size ablation meaningful.

**Flash-decoding (split-K).** The v1 grid is `num_seqs × num_kv_heads` programs. At batch
size 1 with 8 KV heads that is 8 programs on a 108-SM GPU: 93% of the machine idle.
`_paged_attn_decode_split_kernel` splits each sequence's KV across `SPLIT_K` programs,
each producing a partial `(acc, m, l)`, and `_combine_kernel` merges them with the usual
log-sum-exp rescale. `choose_split_k()` picks the split from the grid size and the SM
count, rounded down to a power of two (the combine kernel indexes splits with
`tl.arange`, which requires one).

### Where the bugs live

Every one of these has a dedicated test in `tests/test_paged_attention.py`:

- **Partial last block.** A context length that is not a multiple of `block_size` leaves
  garbage in the tail of the last page. `test_partial_last_block_is_masked` poisons that
  tail and asserts the output does not move.
- **GQA head mapping.** `test_gqa_head_mapping` fills KV head *g* with the constant *g*,
  so the output for query head *h* literally *is* the index of the head it read.
- **fp16 accumulation.** Accumulate in fp32 or the error grows with context length.
  `test_fp32_accumulation_at_long_context` compares the error at 1k and 8k tokens; a
  growing error is the signature.
- **Non-contiguous block tables.** Every fixture shuffles its block ids. A fixture with
  contiguous blocks passes even with a completely broken block-table lookup, which
  defeats the entire purpose of the test.

---

## 5. The step, end to end

```
Scheduler.schedule()              → SchedulerOutput (prefills, decodes, preempted, CoW)
copy_blocks(...)                  → resolve copy-on-write BEFORE anything writes K/V
ModelRunner.prepare_inputs(...)   → one flat batch: [prefill tokens | decode tokens]
AttentionState.plan(...)          → once per step, not once per layer
model.forward(...)                → reshape_and_cache → prefill attn → decode attn
model.compute_logits(...)         → ONLY at logits_indices
Sampler(...)                      → batched; one .tolist() sync per step
```

Two efficiency notes that are easy to get wrong and expensive to leave wrong:

- **Logits are computed only where needed** — the last token of each completed prefill,
  plus every decode. Running a 128k-vocab `lm_head` over all 8192 batched tokens instead
  of the ~64 positions that need one is the largest GEMM in the step, and almost all of
  it would be thrown away.
- **One host→device copy per tensor**, from a pinned buffer, built from flat Python
  lists. A per-sequence `.to(device)` is a sync per sequence and dominates the profile
  at batch 256.

---

## 6. CUDA graphs

At batch size 1 on a 1B model, roughly 16 layers × ~8 kernels each are launched per
step, every one of them a few microseconds of GPU work sitting behind a few microseconds
of launch. The GPU is idle for much of the timeline, and `scripts/profile.py breakdown`
shows it as a low "gpu busy %" without needing nsys installed.

A captured graph replays the entire decode step as one submission. The requirements that
bite, all handled by `CUDAGraphRunner`:

- Every input tensor is preallocated at max shape and reused — a graph captures
  *pointers*, not values.
- No CPU↔GPU sync inside the captured region, which is why `split_k` is pinned to 1
  during capture (the auto-selector reads `context_lens.max()` on the host).
- Padded rows must be *harmless*. `allocate_kv_cache` reserves one extra scratch block
  past the pool the `BlockManager` knows about; padding rows write their garbage K/V
  there and read a context length of 1. That is a branch-free way to make padding safe
  in every backend, and it costs one block.
- Graphs share one memory pool, so capturing 13 buckets does not cost 13 activation
  workspaces.

---

## 7. What's missing

Current limitations:

- **Tensor / pipeline parallelism.** Single GPU only. Adding TP means sharding the
  attention and MLP weights and an NCCL all-reduce per layer; the block manager is
  unaffected, which is a nice property of the design.
- **Quantization.** No AWQ/GPTQ weights, no FP8 KV cache. The KV cache dtype is already
  a config knob (`kv_cache_dtype`), so FP8 KV is mostly a kernel change.
- **MoE.** No expert routing.
- **Beam search / parallel sampling.** The allocator already supports it — `fork()` and
  copy-on-write exist and are tested — but the scheduler and sampler do not expose it.
- **Speculative decoding.** The scheduler would need to accept multi-token proposals per
  step and the sampler an acceptance test.
- **Disaggregated prefill/decode.** Would need KV transfer between engine instances.
- **Multi-modal inputs, LoRA adapters, structured/guided decoding.**
