#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${SWEEP_LOG_DIR:-$REPO_ROOT/runs/_sweep}"
SEEDS="${SWEEP_SEEDS:-3}"

cd "$REPO_ROOT"

python -m pip install --upgrade pip
python -m pip install uv_build numpy pygame tqdm typer
python -m pip install -e . --no-deps

mkdir -p "$LOG_DIR"

cmd=(python -u -m rl_thesis.cli reward-sweep --seeds "$SEEDS" --log-dir "$LOG_DIR")
if [[ -n "${SWEEP_STEPS:-}" ]]; then
  cmd+=(--steps "$SWEEP_STEPS")
fi

if [[ -n "${SWEEP_WORKERS:-}" ]]; then
  cmd+=(--workers "$SWEEP_WORKERS")
fi

if [[ -n "${SWEEP_GPU_SLOTS:-}" ]]; then
  cmd+=(--gpu-slots "$SWEEP_GPU_SLOTS")
fi

if [[ -n "${SWEEP_CONFIGS:-}" ]]; then
  IFS=',' read -r -a configs <<< "$SWEEP_CONFIGS"
  for config in "${configs[@]}"; do
    cmd+=(--config "$config")
  done
fi

nohup "${cmd[@]}" > "$LOG_DIR/coordinator.log" 2>&1 < /dev/null &
echo $! > "$LOG_DIR/coordinator.pid"
echo "Started reward sweep coordinator with PID $(cat "$LOG_DIR/coordinator.pid")"
echo "Coordinator log: $LOG_DIR/coordinator.log"
