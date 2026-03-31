#!/bin/bash
set -euo pipefail

# === Setup ===
cd /workspace
tar xzf rl-thesis-deploy.tar.gz

# Install deps alongside the pre-installed torch from the docker image
pip install --no-deps -e . 2>&1
pip install numpy tqdm typer pygame 2>&1

# Verify CUDA is available
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')"

echo "=== Starting training runs ==="
STEPS=2000000
EVAL_EPS=20
DEMOS=100
mkdir -p runs

# Run 3 seeds in parallel, each logging to its own file
for SEED in 42 43 44; do
    echo "Launching seed $SEED..."
    python -m rl_thesis.cli train \
        --config engineered --seed "$SEED" \
        --steps "$STEPS" --eval-episodes "$EVAL_EPS" --demos "$DEMOS" \
        > "runs/train_seed_${SEED}.log" 2>&1 &
done

echo "=== All 3 seeds launched. Waiting for completion... ==="
wait
echo "=== All training complete ==="

# Show final eval results for each seed
for SEED in 42 43 44; do
    echo ""
    echo "=== Seed $SEED: last 3 evals ==="
    tail -3 "runs/engineered/seed_${SEED}/logs/eval.csv"
done

# Run benchmark on best checkpoints
for SEED in 42 43 44; do
    CKPT="runs/engineered/seed_${SEED}/checkpoints/model_best.pt"
    if [ -f "$CKPT" ]; then
        echo ""
        echo "=== Benchmark: seed $SEED best checkpoint ==="
        python -m rl_thesis.cli benchmark \
            --checkpoint "$CKPT" --config engineered \
            --episodes 100 --start-seed 1000 \
            > "runs/benchmark_seed_${SEED}.log" 2>&1
        cat "runs/benchmark_seed_${SEED}.log"
    fi
done

echo ""
echo "=== All done. Results in /workspace/runs/ ==="
