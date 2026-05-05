# Detecting and Mitigating Query-Based System Prompt Leakage Attacks

## Overview

This repository contains an open-science implementation of AdvGuard, a prompt-level adversarial detection system for large language model (LLM) interactions. It combines:

- prompt level detection using a pretrained text classification model,
- cosine similarity graph construction over incoming prompts for structural detection,
- graph anomaly detection with a GCN-based confirmation stage.

The target use case is detecting adversarially crafted prompts in benign and multi-turn conversational streams.

## Repository structure

- `main.py` — main evaluation pipeline for loading prompt streams, running AdvGuard, and producing detection output.
- `advGuard.py` — core AdvGuard detector implementation, including embedding, similarity graph construction, early prompt-injection scoring, and anomaly detection.
- `environment.yml` — conda environment specification for reproducing the required Python dependencies.
- `run_advguard.sh` — example experiment script showing how to run multiple detection scenarios.
- `data/` — dataset directory with baseline prompts, benign prompt sources, and attack prompt collections.
- `gnnTraining/` — training utilities for the GCN model used by AdvGuard.

## Open science goals

This README is intended to help researchers reproduce the detection experiments and understand the overall pipeline. The design emphasizes transparency through documented data sources, model configuration, and runnable scripts.

## Setup

1. Create the conda environment:

```bash
conda env create -f environment.yml
conda activate defense
```

2. Verify that the required GPU or CPU dependencies are installed and accessible.

3. Ensure `ndss/data/4model/gcn_model.pt` is available or replace `--model_path` with another trained GCN checkpoint.

## Running the detector

Use `main.py` to run a single experiment. Example:

```bash
python main.py \
  --baseline_file ndss/data/1baseline/baseline-prompt-10k.csv \
  --benign_file ndss/data/4multi-turn/1wildChat/wildchat_prompts-1KT.csv \
  --adv_file ndss/data/3attack_prompt/PromptFuzz/Llama-3.3-70B-Instruct/prompt-2k.txt \
  --model_path ndss/data/4model/gcn_model.pt \
  --baseline_size 10000 \
  --ttd 10 \
  --detection_interval 30 \
  --json_output \
  --json_dir ndss/output/PromptFuzz
```

The `run_advguard.sh` script contains several curated example runs for baseline and attack evaluations.

## Data layout

- `data/1baseline/` — baseline prompt corpus used to initialize normal prompt similarity.
- `data/2benign/` — benign prompt datasets from multiple sources.
- `data/3attack_prompt/` — adversarial attack prompt families, including LeakAgent, Pleak, Prompt-Extraction, PromptFuzz, and QROA.
- `data/4multi-turn/` — multi-turn conversation prompts used for stream-based evaluation.
- `data/6groundTruth/` — additional curated prompt examples and ground-truth sources.

## Notes

- `main.py` accepts `.csv` and `.txt` prompt files. For CSV inputs, specify the prompt column with `--benign_column` or `--adv_column`.
- The pipeline is built for reproducible open-science experiments; keep experiment parameters and output directories organized.
- If GPU resources are not available, the code will fall back to CPU for PyTorch operations.

## Reproducibility tips

- Use the fixed random seed option `--seed 42` to reproduce stream shuffling.
- Keep `--baseline_size` consistent when comparing runs.
- Save JSON output with `--json_output` for later analysis.

## Contact

For collaboration or questions, refer to the project maintainers or the corresponding NDSS paper associated with this repository.
