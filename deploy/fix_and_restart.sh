#!/bin/bash
# Run on Vast.ai to clean up and restart training
set -e
cd /workspace/TALH

# Kill any running training
pkill -f "talh.train_torch" 2>/dev/null || true
sleep 2

# Clean corrupt checkpoints
echo "=== Cleaning corrupt checkpoints ==="
rm -rf experiments/medium/checkpoints/talh_full/
rm -f experiments/medium/logs/train_full.csv
rm -f experiments/medium/logs/full_stdout.log
echo "Done"

# Verify fixes
echo ""
echo "=== SSM fix check ==="
grep -c "_ssm_sequential_scan" talh/train_torch.py && echo "Sequential scan: OK"

echo ""
echo "=== LR and grad_clip ==="
grep "lr 1e-4" deploy/train_medium.sh && echo "LR: OK"
grep "grad_clip 0.5" deploy/train_medium.sh && echo "Grad clip: OK"

echo ""
echo "=== Starting training in tmux ==="
tmux kill-session -t train 2>/dev/null || true
tmux new-session -d -s train "bash deploy/train_medium.sh 2>&1 | tee experiments/medium/logs/master.log"
sleep 3
echo "Training started. Checking tmux:"
tmux ls