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
