# paged-inference-engine

A small Llama inference engine with a paged KV cache, continuous batching, and
Triton attention kernels. The allocator and scheduler are plain Python; the model
and attention paths use PyTorch, with optional CUDA backends.

GPU benchmarks have not been published. The scripts are included, but the
scheduler simulator's output is a cost-model estimate, not measured GPU throughput.

## Run locally

```bash
python -m pip install -e ".[dev,cpu]"
make test
python -m bench.scheduler_sim --sweep --num-requests 200 --num-blocks 512
```

CPU tests use a tiny random-weight Llama and compare the paged path against a
cacheless reference. They cover mixed-length batches, chunked prefill, preemption,
and prefix sharing. CUDA tests skip when the hardware is unavailable.

For just the allocator and scheduler:

```bash
python -m pip install -e ".[dev]"
make test-notorch
```

## Generate text

Real model inference needs model weights and the GPU dependencies. Access to the
Llama weights may require accepting the model's license on Hugging Face.

```bash
python -m pip install -e ".[dev,gpu]"
