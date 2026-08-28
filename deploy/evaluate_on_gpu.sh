#!/bin/bash
# ---------------------------------------------------------------
# GPU-side Evaluation — Run BEFORE tearing down the Vast.ai instance
#
# Produces:
#   experiments/medium/results/validation_perplexity.csv
#   experiments/medium/results/generation_samples.txt
#   experiments/medium/results/training_summary.json
#
# Edge-device evaluations (memory, latency, NIAH) are run locally
# on Mac after syncing checkpoints.
#
# Usage:
#   bash deploy/evaluate_on_gpu.sh
# ---------------------------------------------------------------
set -euo pipefail

cd /workspace/TALH

RESULTS_DIR="./experiments/medium/results"
mkdir -p "$RESULTS_DIR"

echo "======================================================"
echo "=== GPU-side Evaluation — $(date) ==="
echo "======================================================"

# ---------------------------------------------------------------
# 1. Validation perplexity for all 5 variants
# ---------------------------------------------------------------
echo ""
echo "--- [1/3] Validation Perplexity ---"

python3 -c "
import csv
import math
import sys
import torch
from pathlib import Path

sys.path.insert(0, '.')
from talh.train_torch import TALHTorch, TrainConfig, TokenisedDataset, _evaluate
from torch.utils.data import DataLoader

results = []
variants = ['full', 'mla_only', 'ssm_only', 'dense_ffn', 'baseline']

for variant in variants:
    ckpt_path = Path(f'experiments/medium/checkpoints/talh_{variant}/best.pt')
    if not ckpt_path.exists():
        print(f'  SKIP {variant} (no best.pt checkpoint)')
        continue

    print(f'  Loading {variant}...')
    ckpt = torch.load(str(ckpt_path), map_location='cuda', weights_only=False)
    raw_cfg = ckpt.get('config', {})
    cfg_kw = {k: v for k, v in raw_cfg.items() if k in TrainConfig.__dataclass_fields__}
    cfg = TrainConfig(**cfg_kw)
    model = TALHTorch(cfg).cuda()
    model.load_state_dict(ckpt['model'])
    model.eval()

    for seq_len in [512, 1024, 2048]:
        try:
            vds = TokenisedDataset('fineweb', split='validation', seq_len=seq_len)
            vl = DataLoader(vds, batch_size=8, num_workers=0)
            val_loss = _evaluate(model, vl, 'cuda', n_batches=100)
            ppl = math.exp(val_loss) if val_loss < 10 else float('inf')
            print(f'  {variant:12s} | seq={seq_len:5d} | val_loss={val_loss:.4f} | ppl={ppl:.2f}')
            results.append({
                'variant': variant,
                'seq_len': seq_len,
                'val_loss': round(val_loss, 4),
                'perplexity': round(ppl, 2),
            })
        except Exception as e:
            print(f'  {variant:12s} | seq={seq_len:5d} | ERROR: {e}')

    del model, ckpt
    torch.cuda.empty_cache()

if results:
    csv_path = '${RESULTS_DIR}/validation_perplexity.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f'  Saved: {csv_path}')
else:
    print('  WARNING: No results produced (no checkpoints found)')
"

# ---------------------------------------------------------------
# 2. Text generation samples (qualitative check)
# ---------------------------------------------------------------
echo ""
echo "--- [2/3] Generation Samples ---"

python3 -c "
import sys
import torch
from pathlib import Path

sys.path.insert(0, '.')
from talh.train_torch import TALHTorch, TrainConfig

try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')
except ImportError:
    print('  WARNING: transformers not installed, skipping generation samples')
    sys.exit(0)

prompts = [
    'Once upon a time, there was a little',
    'The weather today is',
    'In a small village near the mountains',
]

output_lines = []

for variant in ['full', 'dense_ffn', 'mla_only']:
    ckpt_path = Path(f'experiments/medium/checkpoints/talh_{variant}/best.pt')
    if not ckpt_path.exists():
        continue

    ckpt = torch.load(str(ckpt_path), map_location='cuda', weights_only=False)
    raw_cfg = ckpt.get('config', {})
    cfg_kw = {k: v for k, v in raw_cfg.items() if k in TrainConfig.__dataclass_fields__}
    cfg = TrainConfig(**cfg_kw)
    model = TALHTorch(cfg).cuda()
    model.load_state_dict(ckpt['model'])
    model.eval()

    header = f'=== {variant} ==='
    print(f'  {header}')
    output_lines.append(header)

    for prompt in prompts:
        ids = tok.encode(prompt)
        x = torch.tensor([ids], device='cuda')
        with torch.no_grad():
            for _ in range(64):
                out = model(x)
                next_id = out['logits'][:, -1, :].argmax(dim=-1, keepdim=True)
                x = torch.cat([x, next_id], dim=1)
        generated_text = tok.decode(x[0].tolist()[:128])
        print(f'    Prompt: {prompt}')
        print(f'    Output: {generated_text}')
        print()
        output_lines.append(f'Prompt: {prompt}')
        output_lines.append(f'Output: {generated_text}')
        output_lines.append('')

    del model, ckpt
    torch.cuda.empty_cache()

with open('${RESULTS_DIR}/generation_samples.txt', 'w') as f:
    f.write('\n'.join(output_lines))
print(f'  Saved: ${RESULTS_DIR}/generation_samples.txt')
" 2>&1 | tee -a "${RESULTS_DIR}/generation_samples.txt"

# ---------------------------------------------------------------
# 3. Training loss summary (extract from CSV logs)
# ---------------------------------------------------------------
echo ""
echo "--- [3/3] Training Loss Summary ---"

python3 -c "
import csv
import json
from pathlib import Path

logs_dir = Path('experiments/medium/logs')
summary = {}

for log_file in sorted(logs_dir.glob('train_*.csv')):
    variant = log_file.stem.replace('train_', '')
    try:
        rows = list(csv.DictReader(open(log_file)))
    except Exception as e:
        print(f'  {variant:12s} | ERROR reading log: {e}')
        continue

    if not rows:
        print(f'  {variant:12s} | empty log')
        continue

    last = rows[-1]
    val_losses = [float(r.get('val_loss', 999)) for r in rows if r.get('val_loss')]
    best_val = min(val_losses) if val_losses else float('inf')

    summary[variant] = {
        'final_step': last.get('step'),
        'final_train_loss': last.get('train_loss'),
        'best_val_loss': round(best_val, 4),
    }
    print(f'  {variant:12s} | final_step={last.get(\"step\")} | best_val={best_val:.4f}')

out_path = 'experiments/medium/results/training_summary.json'
with open(out_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'  Saved: {out_path}')
"

echo ""
echo "======================================================"
echo "=== GPU evaluation complete — $(date) ==="
echo "======================================================"
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo "Next: sync to Mac with  bash deploy/sync_checkpoints.sh <ip> <port>"
