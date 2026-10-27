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
