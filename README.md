# TALH

TALH is an experimental decoder-only language-model architecture. The current
paper describes TALH as **Adaptive Latent Hybrid** and reports a single-seed
ablation of MLA, a custom minimal recurrent branch, dense FFN, and ternary MoE
components spanning approximately 117--217M active parameters per token.

The paper is an exploratory implementation study, not a claim that TALH is a
production-ready or state-of-the-art model. In the archived run, removing the
recurrent branch hurts validation loss most, and the tested ternary MoE does not
beat the dense FFN. Historical Apple M3 TTFT values are retained as preliminary
observations because raw repeated-trial records and paper-scale checkpoints are
not included.

## Recommended Reading

1. `PROJECT_BRIEFING.md` for the project summary and current conclusions.
2. `paper/talh_150m_ablation.pdf` for the paper.
3. `paper/talh_150m_ablation.tex` for the source text.
4. `replication/README.md` for reproduction-focused instructions.
5. `replication/results/` and `paper/figures/` for supporting result artifacts.

## Repository Layout

```text
TALH/
├── PROJECT_BRIEFING.md
├── paper/
│   ├── talh_150m_ablation.tex
│   ├── talh_150m_ablation.pdf
│   └── figures/
├── replication/
│   ├── README.md
│   ├── configs/
│   ├── results/
│   ├── scripts/
│   └── talh/
├── talh/
├── scripts/
├── configs/
├── deploy/
├── tests/
└── requirements.txt
```

Large model checkpoints, local datasets, logs, credentials, and machine-specific
transfer scripts are intentionally excluded from Git.
