# Early Detection of Model Collapse in LLMs — A Diversity-Based Framework

Reference implementation for the paper *Early Detection of Model Collapse in
Large Language Models: A Diversity-Based Framework* (ICDSA 2026 / Springer
LNNS). It runs recursive fine-tuning experiments (LLaMA 3-8B with QLoRA + GPT-2
Medium; replace and mixed scenarios; three seeds for GPT-2) and analyzes the
per-generation metrics to reproduce the paper's tables and its central claim:
**diversity metrics alert before perplexity.**

## The one rule this code keeps

Nothing is hardcoded to the paper's numbers. Every metric flows from generated
text through `metrics.py`, and detection/lead-time is computed from those live
series in `analyze.py`. The robust finding is the **ordering**; the lead time
the code prints is whatever the data shows (+1 or +2), not a value forced to
match the paper.

## Layout

```
experiments/
  config.py                 CONFIG, corpora, prompts (per-model LR lives here)
  metrics.py                Distinct-n (per-sample), TTR, entropy, vocab,
                            repetition, Self-BLEU, perplexity
  run_experiment.py         load / train / generate; recursive runs; saves JSON
  analyze.py                reads results JSON -> metric panel, detection, LaTeX
  threshold_sensitivity.py  Table 7 sweep over detection thresholds
  colab_notebook.ipynb      end-to-end notebook (live tables + Figure 1)
paper/                      LaTeX source, figures, compiled PDF
results/                    experiment output JSON (gitignored)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

LLaMA 3-8B is gated — accept its license on Hugging Face and authenticate
(`huggingface-cli login`, or set `HF_TOKEN`) before a real run.

## Run

Experiments and analysis are separate steps: `run_experiment.py` does the
training and writes JSON to `results/`; `analyze.py` and
`threshold_sensitivity.py` post-process that JSON.

```bash
# Full run (needs a GPU; an A100 40GB matches the paper)
python experiments/run_experiment.py --model all --scenario both --multi-seed

# GPT-2 only, replace scenario, with Self-BLEU (Table 6)
python experiments/run_experiment.py --model gpt2 --scenario replace --multi-seed --self-bleu

# Analyze + threshold sweep
python experiments/analyze.py --input results/all_results.json
python experiments/threshold_sensitivity.py --input results/all_results.json
```

### CPU self-test (no GPU, no model downloads)

```bash
python experiments/run_experiment.py --dry-run
python experiments/analyze.py --input results/selftest_results.json
python experiments/threshold_sensitivity.py --input results/selftest_results.json
```

`--dry-run` writes a `results/selftest_results.json` built from **deliberately
synthetic toy data** (panels carry `"synthetic": true`, and the perplexity is a
documented toy curve, not a measurement). It exists only to exercise the
analysis plumbing on a laptop — never to produce paper numbers.

## Method notes baked into the code

- **Distinct-n / TTR are per-sample averaged** (the standard definition), so
  Gen-0 Distinct-1 lands ~0.7–0.8. A pooled variant is kept as
  `distinct_n_pooled` / `type_token_ratio_pooled` for reference; if Gen-0 reads
  ~0.19, the pooled definition crept back in and the numbers won't match
  Tables 3–4.
- **TTR equals Distinct-1 by construction** for unigrams (unique tokens / total
  tokens). The columns will match; report only one.
- **GPT-2 uses a lower learning rate (2e-5) than LLaMA's QLoRA (2e-4).** Full
  fine-tuning at 2e-4 collapses GPT-2 in a single generation, leaving no
  resolution to measure a lead. The gentler rate lets the experiment report
  whatever ordering is actually present rather than forcing +0.
- **Two repetition definitions exist:** `repetition_rate` (per-sample
  `1 − unique/total`, the default) and `repetition_rate_consecutive`
  (immediate-repeat fraction). They differ in magnitude; pick one for Table 9
  and state it in the paper.

## Citation

Bisht, K. S. *Early Detection of Model Collapse in Large Language Models: A
Diversity-Based Framework.* ICDSA 2026 / Springer LNNS.
