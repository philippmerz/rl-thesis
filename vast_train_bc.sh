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

echo "=== Starting BC+RL training runs ==="
STEPS=2000000
EVAL_EPS=20
DEMOS=100
BC_EPS=200
EPS_START=0.1
mkdir -p runs

for SEED in 42 43 44; do
    echo "Launching BC+RL seed $SEED..."
    python -m rl_thesis.cli train \
        --config engineered --seed "$SEED" \
        --steps "$STEPS" --eval-episodes "$EVAL_EPS" \
        --demos "$DEMOS" --bc-episodes "$BC_EPS" \
        --epsilon-start "$EPS_START" \
        --lr-schedule constant \
        > "runs/train_bc_seed_${SEED}.log" 2>&1 &
done

echo "=== All 3 seeds launched. Waiting for completion... ==="
wait
echo "=== All training complete ==="

for SEED in 42 43 44; do
    echo ""
    echo "=== BC+RL Seed $SEED: last 5 evals ==="
    tail -5 "runs/engineered/seed_${SEED}/logs/eval.csv"
done

for SEED in 42 43 44; do
    CKPT="runs/engineered/seed_${SEED}/checkpoints/model_best.pt"
    if [ -f "$CKPT" ]; then
        echo ""
        echo "=== Benchmark: BC+RL seed $SEED ==="
        python -m rl_thesis.cli benchmark \
            --checkpoint "$CKPT" --config engineered \
            --episodes 100 --start-seed 1000 \
            > "runs/benchmark_bc_seed_${SEED}.log" 2>&1
        cat "runs/benchmark_bc_seed_${SEED}.log"
    fi
done

echo ""
echo "=== All done ==="
