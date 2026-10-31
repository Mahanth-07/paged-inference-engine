#!/usr/bin/env bash
# Provision a freshly rented GPU box. Idempotent -- safe to re-run.
#   bash scripts/setup_gpu_box.sh
#
# Money-saving notes:
#   * Put HF_HOME on a persistent network volume so you download Llama once, not
#     once per session. On RunPod that's /workspace; on Vast it's whatever you
#     mounted. 15GB re-downloaded every session at $2/hr adds up fast.
#   * TERMINATE the pod when done, don't just stop it, unless you're deliberately
#     paying to keep the volume.
#   * Set a billing alert. Everyone forgets a box running overnight exactly once.
set -euo pipefail

VOLUME=${VOLUME:-/workspace}
export HF_HOME="$VOLUME/hf"
mkdir -p "$HF_HOME"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "=== deps ==="
pip install -q --upgrade pip
pip install -e ".[dev,gpu,bench]"

echo "=== versions (record these in BENCHMARKS.md for reproducibility) ==="
python - <<'PY'
import torch, importlib
print(f"torch          {torch.__version__}  cuda {torch.version.cuda}")
print(f"device         {torch.cuda.get_device_name(0)}")
p = torch.cuda.get_device_properties(0)
print(f"sm             {p.major}.{p.minor}   SMs {p.multi_processor_count}   VRAM {p.total_memory/2**30:.1f} GiB")
for m in ("triton", "flashinfer", "flash_attn", "vllm", "transformers"):
    try:
        print(f"{m:<14} {importlib.import_module(m).__version__}")
    except Exception as e:
        print(f"{m:<14} MISSING ({type(e).__name__})")
PY

echo "=== peak HBM bandwidth (your memory-bound roofline ceiling) ==="
python - <<'PY'
import torch, time
x = torch.empty(1 << 28, dtype=torch.float16, device="cuda")  # 512 MiB
y = torch.empty_like(x)
for _ in range(3):
    y.copy_(x)
torch.cuda.synchronize()
t = time.perf_counter()
N = 50
for _ in range(N):
    y.copy_(x)
torch.cuda.synchronize()
dt = time.perf_counter() - t
gbs = 2 * x.numel() * 2 * N / dt / 1e9  # read + write
print(f"measured copy bandwidth: {gbs:.0f} GB/s")
print("Report your decode kernel's achieved bandwidth as a fraction of this number.")
PY

echo "=== prefetch weights to the persistent volume ==="
MODELS=${MODELS:-"meta-llama/Llama-3.2-1B-Instruct"}
for m in $MODELS; do
  python -c "
from huggingface_hub import snapshot_download
snapshot_download('$m', allow_patterns=['*.safetensors', '*.json', '*.model', 'tokenizer*'])
print('cached $m')
"
done

echo "=== ShareGPT trace (the benchmark dataset vLLM itself uses) ==="
# Onto the persistent volume too, and symlinked into the repo: 650MB re-downloaded
# every session at $2/hr is exactly the kind of waste this script exists to prevent.
DATA_DIR="$VOLUME/data"
mkdir -p "$DATA_DIR" data
SHAREGPT="$DATA_DIR/ShareGPT_V3_unfiltered_cleaned_split.json"
if [ ! -f "$SHAREGPT" ]; then
  curl -L -o "$SHAREGPT" \
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
fi
ln -sf "$SHAREGPT" data/ShareGPT_V3_unfiltered_cleaned_split.json
echo "sharegpt: $(du -h "$SHAREGPT" | cut -f1)"

echo "=== smoke test ==="
python -m pytest tests/ -q -x
python -c "
import torch
from paged_inference_engine.config import EngineConfig
from paged_inference_engine.engine import LLMEngine
from paged_inference_engine.sampling import SamplingParams
e = LLMEngine(EngineConfig.from_yaml('configs/default.yaml'))
o = e.generate(prompts=['The capital of France is'], sampling_params=SamplingParams(temperature=0.0, max_tokens=16))
print('generated:', repr(o[0].text))
"

echo
echo "done. HF_HOME=$HF_HOME"
echo "next:  make test-gpu      # the correctness oracle against HF"
echo "       make bench         # the full sweep, ~2h unattended"
