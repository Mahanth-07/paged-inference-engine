#!/usr/bin/env bash
# Full benchmark sweep. Run once, unattended, on a rented A100.
# Produces results/*.json which bench/plot.py turns into the graph at the top of the README.
#
#   bash bench/run_sweep.sh 2>&1 | tee results/sweep.log
#
# Roughly 2h at the defaults. Everything is matched across systems on purpose: same
# model, dtype, max-model-len, token budget, seq cap, memory fraction, and the SAME
# client process. Record any configuration differences with the results.
set -euo pipefail

MODEL=${MODEL:-meta-llama/Llama-3.2-1B-Instruct}
RATES=${RATES:-"1 2 4 8 12 16 24 32 inf"}
NUM_PROMPTS=${NUM_PROMPTS:-500}
REPEATS=${REPEATS:-3}
MAX_BATCHED=${MAX_BATCHED:-8192}
MAX_SEQS=${MAX_SEQS:-256}
GPU_UTIL=${GPU_UTIL:-0.90}
MAX_LEN=${MAX_LEN:-4096}
DTYPE=${DTYPE:-bfloat16}
BACKEND=${BACKEND:-flashinfer}
DATASET=${DATASET:-sharegpt}
DATASET_PATH=${DATASET_PATH:-data/ShareGPT_V3_unfiltered_cleaned_split.json}
mkdir -p results docs/figures

echo "=== environment (record this in BENCHMARKS.md) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python - <<'PY'
import importlib, torch
print(f"torch {torch.__version__} cuda {torch.version.cuda}")
for m in ("triton", "flashinfer", "flash_attn", "vllm", "transformers"):
    try:
        print(f"{m} {importlib.import_module(m).__version__}")
    except Exception:
        print(f"{m} MISSING")
PY

wait_for_server() {
  for _ in $(seq 1 180); do
    curl -s "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "server on port $1 never came up" >&2; exit 1
}

kill_tree() { kill "$1" 2>/dev/null || true; wait "$1" 2>/dev/null || true; sleep 5; }

bench_system() {           # $1=name  $2=port
  local name=$1 port=$2
  for rep in $(seq 1 "$REPEATS"); do
    for rate in $RATES; do
      echo "=== $name  rate=$rate  rep=$rep ==="
      python -m bench.client \
        --backend "$name" --port "$port" --model "$MODEL" \
        --dataset "$DATASET" --dataset-path "$DATASET_PATH" \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate "$rate" --seed "$rep" \
        --out "results/${name}_rate${rate}_rep${rep}.json"
    done
  done
}
