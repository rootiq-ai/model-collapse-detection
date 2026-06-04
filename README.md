# Early Detection of Model Collapse in LLMs — A Diversity-Based Framework

Reference implementation for the paper *Early Detection of Model Collapse in
Large Language Models: A Diversity-Based Framework*. It runs the recursive
fine-tuning experiments (LLaMA 3-8B with QLoRA + GPT-2 Medium, replace and
mixed scenarios, three seeds) and regenerates **every paper table (1–9), the
cross-architecture comparison table, and Figure 1 from the live run**.

## The one rule this code keeps

Nothing in the output path is hardcoded from the paper. The published numbers
live only in [`mcd/reference.py`](mcd/reference.py) (`PAPER_REF`) and are used
solely for an on-screen drift comparison and an ordering check — they never
feed a table or the figure. The robust claim is the **ordering** (diversity
metrics alert before perplexity); the **lead time the code prints is whatever
the live data shows** (+1 or +2), not a value baked in to match the paper.

## What reproduces, and what doesn't

A genuine recursive fine-tuning run reproduces (a) the framework and the
table/figure structure with the same metric columns, and (b) — with the
learning-rate calibration in `CONFIG` — the ordering and the early-warning
lead. Exact decimals depend on hardware, library versions, and sampling RNG,
and are expected to drift. The 10% threshold and 10-sample seed corpus are
deliberate experimental choices that accelerate collapse for controlled study;
they require recalibration before production use.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

LLaMA 3-8B is gated — accept its license on Hugging Face and authenticate
(`huggingface-cli login`, or set `HF_TOKEN`) before a real run.

## Run

```bash
# Full run: LLaMA 3-8B + GPT-2 (needs a GPU; an A100 40GB matches the paper)
python -m mcd.run

# GPT-2 only (skip the gated LLaMA download)
python -m mcd.run --models gpt2

# CPU pipeline self-test — no GPU, no model downloads
python -m mcd.run --dry-run
```

Useful overrides: `--generations N`, `--seeds 42,123,456`, `--threshold 10`,
`--output-dir outputs`.

### The `--dry-run` self-test

`--dry-run` feeds **deliberately synthetic toy data** through the
metrics → detection → tables → figure pipeline so you can verify the analysis
code on a laptop without a GPU. Every artifact it writes is stamped
`PIPELINE SELF-TEST — NOT EXPERIMENTAL RESULTS`, and the perplexity it uses is a
documented toy curve, not a model measurement. It is for checking the plumbing,
never for producing paper numbers.

## Outputs

Written to `outputs/` (override with `--output-dir`):

- `table1_lead_time.tex` … `table9_degradation.tex`, `comparison_table.tex`
- `figure1_detection_timeline.png`
- `experiment_results_complete.json` (metrics, CIs, detection, threshold sweep)

## Layout

```
mcd/
  config.py        CONFIG + seeding (per-model learning rate lives here)
  corpus.py        10 training texts + 5 held-out test texts
  metrics.py       Distinct-n (per-sample), TTR, entropy, vocab, repetition,
                   Self-BLEU, perplexity
  models.py        load_llama3 (QLoRA 4-bit), load_gpt2
  experiment.py    generate / train / recursive run + bootstrap CI aggregation
  detection.py     threshold crossing, sweep (Table 7), reporting, comparison
  figures.py       Figure 1 (detection timeline)
  latex_tables.py  Tables 1–9 + comparison, emitted from live results
  reference.py     PAPER_REF (comparison only) + ordering check
  synthetic.py     toy data for the --dry-run self-test
  run.py           CLI entry point
```

## Method notes baked into the code

- **Distinct-n / TTR are per-sample averaged** (the standard definition), so
  Gen-0 Distinct-1 lands ~0.7–0.8. A pooled variant is kept as
  `distinct_n_pooled` for reference only; if Gen-0 prints ~0.19 the pooled
  definition crept back in.
- **TTR equals Distinct-1 by construction** for unigrams (unique tokens / total
  tokens). The columns will match; report only one.
- **GPT-2 uses a lower learning rate (2e-5) than LLaMA's QLoRA (2e-4).** Full
  fine-tuning at 2e-4 collapses GPT-2 in a single generation, leaving no
  resolution to measure a lead. The gentler rate lets the experiment report
  whatever ordering is actually present rather than forcing +0.
- **`repetition_rate`** is `1 − unique/total` per sample; recalibrate to the
  paper's definition on rerun if the magnitude differs.

## Citation

Bisht, K. S. *Early Detection of Model Collapse in Large Language Models: A
Diversity-Based Framework.* ICDSA 2026 / Springer LNNS.
