# Model Collapse Detection Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> **Early Detection of Model Collapse in Large Language Models: A Diversity-Based Framework**
> Kamal Singh Bisht
> *ICDSA 2026 — Proceedings published in Springer Lecture Notes in Networks and Systems (LNNS), Scopus-indexed*
> ORCID: [0009-0006-9706-1572](https://orcid.org/0009-0006-9706-1572)

## The Core Finding: Ordering > Thresholds

**Diversity metrics detect collapse before perplexity.**

This *ordering* is the robust finding. Specific thresholds (10%) and lead times (≈2 generations) are experimental artifacts of our setting that require recalibration before production use. The ordering held across every condition we tested.

| Condition                       | Ordering held? |
|---------------------------------|----------------|
| Thresholds 5–20%                | ✅ Yes |
| Scales 355M (GPT-2) and 8B (LLaMA-3) | ✅ Yes |
| Contamination 30% and 100%      | ✅ Yes |
| Self-BLEU comparison (Table 6)  | ✅ Yes |

**Trust the ordering. Calibrate the thresholds.**

## What This Repo Reproduces

This repository contains the code and seed/test corpora for the ICDSA 2026 paper. It reproduces the experiments reported in the manuscript:

- Replace and mixed (30% synthetic) recursive fine-tuning on **GPT-2 Medium** (full fine-tuning) and **LLaMA 3-8B** (QLoRA 4-bit).
- The full Tier 1 / Tier 2 / Tier 3 metric panel (Distinct-1/2/3, TTR, token entropy, perplexity, vocabulary coverage, repetition rate).
- Multi-seed GPT-2 runs (seeds 42, 123, 456) with bootstrap 95% confidence intervals.
- Self-BLEU comparison (Table 6) for the Distinct-1 vs Self-BLEU efficiency analysis.
- Threshold sensitivity sweep (Table 7) over {5%, 10%, 15%, 20%}.
- Detection-timing analysis that emits both a human-readable summary and LaTeX rows for the paper's tables.

The repo intentionally does **not** include WikiText-103 / code-generation / tipping-point experiments — those are scoped as future work in the paper.

## Generation Mapping (Table 1 in the paper)

| Your retraining cadence | 1 generation = | 2-generation lead = |
|-------------------------|----------------|---------------------|
| Quarterly refresh       | 3 months       | **6 months of warning** |
| Monthly retraining      | 1 month        | **2 months of warning** |
| Weekly fine-tuning      | 1 week         | **2 weeks of warning**  |

## Repository Structure

```
model-collapse-detection/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── experiments/
│   ├── __init__.py
│   ├── config.py                   # Hyperparameters + 10-sample seed corpus + 5-sample test corpus
│   ├── metrics.py                  # All diversity / perplexity / Self-BLEU metrics
│   ├── run_experiment.py           # Recursive fine-tuning (replace + mixed; multi-seed)
│   ├── analyze.py                  # Detection timing, lead time, LaTeX-ready table rows
│   ├── threshold_sensitivity.py    # Reproduces Table 7 (post-hoc, no retraining)
│   └── colab_notebook.ipynb        # One-click runner
├── paper/
│   └── icdsa2026_model_collapse.tex
└── results/
    └── .gitkeep
```

## Quick Start

### Option 1 — Google Colab (recommended)

1. Upload `experiments/colab_notebook.ipynb` to Google Colab.
2. Enable GPU: `Runtime → Change runtime type → A100` (or T4 for GPT-2-only).
3. Run all cells. Full run (LLaMA + GPT-2 multi-seed) is ~10–12 h on A100. GPT-2-only is ~3 h.

### Option 2 — Local

```bash
git clone https://github.com/rootiq-ai/model-collapse-detection.git
cd model-collapse-detection
pip install -r requirements.txt

# Full reproduction: both models, both scenarios, multi-seed GPT-2, Self-BLEU
python experiments/run_experiment.py --model all --scenario both --multi-seed --self-bleu --output results

# Faster: GPT-2 replace scenario only, single seed
python experiments/run_experiment.py --model gpt2 --scenario replace --output results

# After the runs finish:
python experiments/analyze.py --input results/all_results.json
python experiments/threshold_sensitivity.py --input results/all_results.json
```

## Detection Framework (Table 2 in the paper)

| Tier        | Metrics                       | Threshold | Action               |
|-------------|-------------------------------|-----------|----------------------|
| 1 (Early)   | Distinct-1, TTR, Entropy      | −10%      | Investigate sources  |
| 2 (Confirm) | Perplexity                    | +10%      | Halt pipeline        |
| 3 (Severity)| Vocabulary coverage           | −20%      | Full remediation     |

## Published Results

The paper's results tables (Tables 3, 4, 6, 7, 8, 9) are produced by running the scripts above. After your run completes, `analyze.py` prints exact LaTeX rows you can paste into the manuscript so that the values reported in the paper match the values in this repo.

To reproduce the headline GPT-2 multi-seed replace-scenario result (paper Table 4):

```bash
python experiments/run_experiment.py --model gpt2 --scenario replace --multi-seed --output results
python experiments/analyze.py --input results/all_results.json
```

## Citation

```bibtex
@inproceedings{bisht2026collapse,
  title     = {Early Detection of Model Collapse in Large Language Models:
               A Diversity-Based Framework},
  author    = {Bisht, Kamal Singh},
  booktitle = {Proceedings of the 7th International Conference on Data Science
               and Applications (ICDSA 2026)},
  series    = {Lecture Notes in Networks and Systems},
  publisher = {Springer},
  year      = {2026}
}
```

## Contact

Kamal Singh Bisht
Email: reachbisht7@gmail.com
ORCID: [0009-0006-9706-1572](https://orcid.org/0009-0006-9706-1572)

## License

MIT License (see `LICENSE`).
