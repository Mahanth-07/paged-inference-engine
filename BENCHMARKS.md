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
