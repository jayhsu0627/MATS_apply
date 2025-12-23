# Analogy Making Can Be Your Choice: Controlling Explanation Style Through Activation Steering

**One-line summary**: We show that a single activation direction, computed from literal vs. analogy explanations, can reliably turn analogy-making on or off in Gemma-2-9B-IT and Llama-3.1-8B-Instruct.

## What this repo contains

- `scripts/` – code to:
  - generate a paired literal/analogy dataset
  - compute analogy steering vectors
  - run layer stability + ablation
  - orthogonalize away lexical triggers (e.g. “imagine”)
  - run sanity checks and LLM-as-Judge evaluation
- `data/` – synthetic paired prompts
- `models/` – saved steering vectors and norms (`.pt`)
- `results/` – JSON analyses (stability, ablation, sanity checks, judge scores)
- `figures/` – main plots
- `doc/` – write-up and planning notes

## How to run (high level)

From the repo root:

```bash
python scripts/generate_data.py
python scripts/compute_vector_multi.py
python scripts/layer_stability_analysis.py
python scripts/layer_ablation.py
python scripts/orthogonalize_vector.py
python scripts/sanity_checks.py
python scripts/llm_judge_eval.py
```

Scripts currently assume this directory layout and (in a few places) absolute paths to `/mnt/drive_b/MATS_apply/`.

## Where to look

- **Main story + results**: `doc/INTERIM_REPORT.md`
- **Experimental design rationale**: “Experimental Design” section in `doc/INTERIM_REPORT.md`
- **Key plots**: `figures/` (stability, ablation, sanity checks, LLM-as-Judge)

If you only have a few minutes, skimming `doc/INTERIM_REPORT.md` is the best entry point.


