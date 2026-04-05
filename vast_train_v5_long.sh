#!/bin/bash
set -euo pipefail

cd /workspace
if [ ! -d "rl-thesis" ]; then
    git clone --branch engineered-config https://github.com/philippmerz/rl-thesis.git
fi
cd rl-thesis
git pull --ff-only

pip install --no-deps -e . 2>&1
pip install numpy tqdm typer pygame 2>&1

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')"

echo "=== Starting V5 long training (5M steps) ==="
EVAL_EPS=20
DEMOS=100
mkdir -p runs

for SEED in 42 43 44; do
    echo "Launching v5_long seed $SEED..."
    python -m rl_thesis.cli train \
        --config engineered_v5_long --seed "$SEED" \
        --eval-episodes "$EVAL_EPS" --demos "$DEMOS" \
        > "runs/train_v5_long_seed_${SEED}.log" 2>&1 &
done

echo "=== All 3 seeds launched. Waiting for completion... ==="
wait
echo "=== All training complete ==="

for SEED in 42 43 44; do
    echo ""
    echo "=== v5_long Seed $SEED: last 5 evals ==="
    tail -5 "runs/engineered_v5_long/seed_${SEED}/logs/eval.csv"
done

for SEED in 42 43 44; do
    CKPT="runs/engineered_v5_long/seed_${SEED}/checkpoints/model_best.pt"
    if [ -f "$CKPT" ]; then
        echo ""
        echo "=== Benchmark: v5_long seed $SEED ==="
        python -m rl_thesis.cli benchmark \
            --checkpoint "$CKPT" --config engineered_v5_long \
            --episodes 100 --start-seed 1000 \
            > "runs/benchmark_v5_long_seed_${SEED}.log" 2>&1
        cat "runs/benchmark_v5_long_seed_${SEED}.log"
    fi
done

echo ""
echo "=== All done ==="
