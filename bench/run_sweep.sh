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
