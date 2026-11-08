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
paged-inference-engine generate --prompt "The capital of France is" --max-tokens 32
```

The Python API exposes the request queue and step loop:

```python
from paged_inference_engine import (
    EngineConfig,
    LLMEngine,
    ModelConfig,
    SamplingParams,
)

engine = LLMEngine(EngineConfig(
    model=ModelConfig(model="meta-llama/Llama-3.2-1B-Instruct"),
))
engine.add_request(
    prompt="The capital of France is",
    sampling_params=SamplingParams(temperature=0.0, max_tokens=32),
)
while engine.has_unfinished_requests():
    for output in engine.step():
        print(output.delta_text, end="", flush=True)
```

## Serve

```bash
python -m paged_inference_engine.entrypoints.api_server \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --port 8000 --backend flashinfer --cuda-graphs
```

The server implements OpenAI-compatible completions and chat completions, including
SSE streaming. It also exposes `/health`, `/v1/models`, and `/metrics`.

## How it fits together

- `block_manager.py` owns physical cache blocks, reference counts, copy-on-write,
  and prefix-cache eviction.
- `scheduler.py` selects work for each forward pass under token and block budgets.
  It uses FCFS admission, chunked prefill, and recompute-based preemption.
- `model_runner.py` builds tensors, manages KV storage, and captures CUDA graphs.
- `models/llama.py` implements the model; `models/reference.py` provides the
  cacheless correctness reference.
- `layers/attention.py` selects naive, FlashInfer, flash-attn, or Triton attention.
- `kernels/` contains paged prefill/decode attention and fused elementwise ops.

The scheduler must never admit work the block manager cannot allocate. Prefixes
are shared only after their KV has been computed, and block hashes include the
preceding context. More detail is in [the design notes](docs/DESIGN.md).

## Benchmarks

`bench/client.py` sends Poisson-arrival requests to this server or vLLM.
`bench/baselines/hf_generate.py` runs Hugging Face static-batch baselines.
The sweep varies request rate and reports throughput, tail latency, and goodput.

On a configured GPU machine:

```bash
python -m pip install -e ".[dev,gpu,bench]"
make test-gpu
make bench
```

The sweep writes results under `results/` and figures under `docs/figures/`.
See [BENCHMARKS.md](BENCHMARKS.md) for the run configuration, metrics, and
ablations. Record the actual hardware and software versions with each run.

## Limits

This is a single-device engine. It does not implement tensor parallelism,
quantization, MoE, speculative decoding, LoRA, or guided decoding. The allocator
supports forking, but beam search is not exposed by the scheduler or sampler.
GPU kernel correctness and performance still need validation on the target hardware.

## References

- [vLLM](https://github.com/vllm-project/vllm) and the PagedAttention paper
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [FlashInfer KV-cache layout](https://docs.flashinfer.ai/tutorials/kv_layout.html)
- Orca, SARATHI, FlashAttention-2, and Flash-Decoding
