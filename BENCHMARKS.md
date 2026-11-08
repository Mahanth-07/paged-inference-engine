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
