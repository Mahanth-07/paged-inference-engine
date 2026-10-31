#!/usr/bin/env bash
# Push the local tree to a rented box, pull results back.
#
#   export BOX="root@1.2.3.4 -p 40123"        # RunPod/Vast give you host+port
#   bash scripts/sync.sh                       # push
#   bash scripts/sync.sh pull                  # pull results/ and docs/figures/
#
# The workflow this enables, and the reason it matters on your setup: write code all
# week on the MacBook, run `make test-cpu` and `make sim-sweep` locally for free, then
# rent a box for one focused session, sync, and run everything you queued. Do not
# develop interactively on a $2/hr box.
set -euo pipefail

: "${BOX:?set BOX, e.g. export BOX='root@1.2.3.4 -p 40123'}"
REMOTE_DIR=${REMOTE_DIR:-/workspace/paged-inference-engine}

# macOS ships BSD grep, which has no -P. Parse with awk so this runs on the MacBook
# you actually develop on, not just on Linux.
# shellcheck disable=SC2086
HOST=$(echo $BOX | awk '{print $1}')
# shellcheck disable=SC2086
PORT=$(echo $BOX | awk '{for(i=1;i<NF;i++) if($i=="-p") {print $(i+1); exit}} END{}')
PORT=${PORT:-22}
SSH="ssh -p $PORT -o StrictHostKeyChecking=no"

EXCLUDES=(--exclude '.git' --exclude '__pycache__' --exclude '*.egg-info'
          --exclude '.pytest_cache' --exclude 'data' --exclude '.venv'
          --exclude 'results' --exclude '*.nsys-rep' --exclude '*.ncu-rep')

if [[ "${1:-push}" == "pull" ]]; then
  rsync -avz -e "$SSH" "$HOST:$REMOTE_DIR/results/" ./results/
  rsync -avz -e "$SSH" "$HOST:$REMOTE_DIR/docs/figures/" ./docs/figures/
  echo "pulled results"
else
  $SSH "$HOST" "mkdir -p $REMOTE_DIR"
  rsync -avz --delete "${EXCLUDES[@]}" -e "$SSH" ./ "$HOST:$REMOTE_DIR/"
  echo "pushed to $HOST:$REMOTE_DIR"
  echo "  $SSH $HOST 'cd $REMOTE_DIR && make test'"
fi
