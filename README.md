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
