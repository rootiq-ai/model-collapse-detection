# Model Collapse Detection Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> **Early Detection of Model Collapse in Large Language Models: A Diversity-Based Framework**  
> Kamal Singh Bisht, *Senior Member, IEEE*  
> IEEE WCCI 2026 (IJCNN Track)

## The Core Finding: Ordering > Thresholds

**Diversity metrics detect collapse before perplexity.**

This *ordering* is the robust finding. Specific thresholds (10%) and lead times (2 generations) are experimental artifacts that require recalibration. The ordering held across every condition we tested.

| Condition | Ordering Held? |
|-----------|----------------|
| Thresholds 5-20% | ✅ Yes |
| Scales 355M-8B | ✅ Yes |
| Contamination 30-100% | ✅ Yes |
| Self-BLEU comparison | ✅ Yes |

**Trust the ordering. Calibrate the thresholds.**

## Narrative Example: Incident Response Timeline

> A company retrains their customer service LLM **monthly**. In **Month 3**, Distinct-1 drops 12%—Tier 1 alert triggered. Investigation reveals 25% of training data is the model's own prior responses. By **Month 4**, provenance filtering implemented. In **Month 5**, perplexity would have finally crossed threshold—but remediation is already complete.

## Generation Mapping

| Your Cadence | 1 Gen = | 2-Gen Lead = |
|--------------|---------|--------------|
| Quarterly | 3 months | **6 months warning** |
| Monthly | 1 month | **2 months warning** |
| Weekly | 1 week | **2 weeks warning** |

## Contact

Kamal Singh Bisht  
Email: reachbisht7@gmail.com

## Repository Structure

```
model-collapse-detection/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── experiments/
│   ├── run_experiment.py          # Main experiment (multi-seed, mixed-ratio)
│   ├── metrics.py                 # Diversity & perplexity metrics
│   ├── config.py                  # Configuration
│   └── colab_notebook.ipynb       # Google Colab notebook
├── paper/
│   └── wcci2026_model_collapse.tex
└── results/
    └── .gitkeep
```

## Quick Start

### Option 1: Google Colab (Recommended)

1. Upload `experiments/colab_notebook.ipynb` to Google Colab
2. Enable A100 GPU: `Runtime → Change runtime type → A100`
3. Run all cells (~10-12 hours)

### Option 2: Local

```bash
git clone https://github.com/anonymous/model-collapse-detection.git
cd model-collapse-detection
pip install -r requirements.txt

# Run all experiments (replace + mixed, multi-seed)
python experiments/run_experiment.py --model all --scenario both --multi-seed
```

## Results Summary

### GPT-2 Medium (Replace, 3 Seeds, 95% CI)

| Gen | D-1 | PPL | ΔD-1 | ΔPPL |
|-----|-----|-----|------|------|
| 0 | 0.756±0.018 | 24.2±0.8 | -- | -- |
| 1 | 0.698±0.021 | 24.8±0.9 | -8±2% | +2±1% |
| 2 | 0.641±0.024 | 25.9±1.1 | **-15±3%** | +7±2% |
| 3 | 0.562±0.028 | 28.4±1.4 | -26±4% | +17±3% |
| 4 | 0.458±0.031 | 34.2±1.8 | -39±4% | **+41±5%** |
| 5 | 0.342±0.035 | 43.7±2.3 | -55±5% | +81±7% |

### Detection Framework

| Tier | Metrics | Threshold | Action |
|------|---------|-----------|--------|
| 1 (Early) | Distinct-1, TTR | -10% | Investigate |
| 2 (Confirm) | Perplexity | +10% | Halt pipeline |
| 3 (Severity) | Entropy, KL | -20% | Remediate |

## Citation

```bibtex
@inproceedings{anonymous2026collapse,
  title={A Unified Framework for Model Collapse in Generative Neural Networks},
  author={Anonymous},
  booktitle={IEEE World Congress on Computational Intelligence (WCCI)},
  year={2026}
}
```

## License

MIT License
