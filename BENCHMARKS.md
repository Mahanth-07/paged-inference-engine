# Benchmarks

No GPU results are recorded here yet. This document describes how to run the
comparison and what to keep with the results.

## Run the sweep

On a GPU machine with the model weights and benchmark dependencies installed:

```bash
make test-gpu
make bench
```

For a shorter harness check:

```bash
make bench-quick
```

The sweep writes JSON files and a log under `results/`. Plotting writes figures
under `docs/figures/` and a summary to `results/summary_table.md`:

```bash
python -m bench.plot --results results --out docs/figures
```

Keep the git revision, GPU model, driver and CUDA versions, library versions,
model, dtype, and cache configuration with each run. The sweep prints environment
information at startup; check that the log captures it.

## Default configuration

| Setting | Value |
| --- | --- |
| Model | meta-llama/Llama-3.2-1B-Instruct |
| Dtype | bfloat16 |
| Maximum model length | 4096 |
| Maximum batched tokens | 8192 |
| Maximum sequences | 256 |
| GPU memory utilization | 0.90 |
| Requests per rate | 500 |
| Repeats | 3 |
| Request rates | 1, 2, 4, 8, 12, 16, 24, 32, infinity |

Check `bench/run_sweep.sh` for overrides and backend settings. Match model, dtype,
length limits, token budget, sequence limit, and memory fraction across servers.
Report any settings that cannot be matched.

## Workload and timing

The client uses Poisson arrivals and ShareGPT prompt lengths. Fixed output lengths
and `ignore_eos` keep token counts comparable. Warmup requests are excluded.
Use the same seed and prompts for each system, and report repeat variation.

TTFT starts at request arrival and ends at the first chunk carrying token text.
Metadata-only SSE chunks do not count. Streaming behavior can affect this
measurement, so check the response format when adding another server.

| Metric | Meaning |
| --- | --- |
| TTFT | Arrival to first token |
| TPOT | Time after first token divided by remaining output tokens |
| ITL | Time between output tokens |
| E2EL | Arrival to completion |
| Output throughput | Output tokens per second over the run |
| Goodput | Requests per second meeting both latency limits |

The default goodput limits are TTFT <= 500 ms and TPOT <= 50 ms.
Report throughput alongside p99 latency: a saturated server can have high
throughput while requests spend most of their time waiting.

## Baselines

- `vllm`: vLLM's server with matched configuration.
- `hf_static`: Hugging Face generate with static batching.
- `hf_compile`: static cache with torch.compile.
- `paged-inference-engine`: this engine.

The Hugging Face runners are offline baselines. Their arrival and queueing model
differs from the HTTP server runs; explain that difference when comparing them.

## Ablations

The sweep includes block sizes 8, 16, 32, and 64; chunked prefill; CUDA graphs;
prefix caching; and a one-block-per-sequence configuration to examine paging.
Use the same prompts and request rate within each comparison.

Prefix-cache hit rates depend on request order and when completed blocks become
available for sharing. Record the shared-prefix workload and cold-start behavior.

## Kernel measurements

```bash
python -m bench.microbench_attention
python -m bench.microbench_attention --prefill
python -m bench.microbench_kernels
python scripts/profile.py breakdown
```

Attention microbenchmarks compare backends across batch and context sizes.
Record measured bandwidth as well as latency. Split-K can help at small batches,
but its reduction overhead may outweigh the benefit at larger batches.

For profiling:

```bash
nsys profile -o results/decode --force-overwrite true \
    python scripts/profile.py timeline --batch 1 --steps 50
ncu --set full -k _paged_attn_decode -o results/decode_ncu \
    python scripts/profile.py kernel --batch 1
```

Look for CPU input-preparation time, gaps between kernel launches, synchronization,
and attention bandwidth. Record observations from the trace before attributing a
performance gap to a particular component.

## Simulator

`bench/scheduler_sim.py` runs the allocator and scheduler with estimated model
latencies. It is useful for exploring memory pressure and scheduling policy.
Its throughput is a simulation result and should be labeled separately from GPU
measurements.
