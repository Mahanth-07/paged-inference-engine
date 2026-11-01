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
