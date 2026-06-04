"""
Metrics for Model Collapse Detection
====================================

All metrics are computed directly from generated text and/or the model.
Nothing is hardcoded to expected values — whatever the experiment produces
flows through these functions.

Tier 1 (early warning):  distinct_n, type_token_ratio, token_entropy
Tier 2 (confirmation):   compute_perplexity
Tier 3 (severity):       vocabulary_coverage, repetition_rate, vocabulary_size
Comparison baseline:     self_bleu  (O(n^2), used only for Table 6 comparison)

DEFINITION NOTE (matches colab_notebook.ipynb)
----------------------------------------------
Distinct-n and TTR are computed PER SAMPLE and AVERAGED. This is the standard
Distinct-n definition and is what reproduces the paper's scale (Gen-0
Distinct-1 ~ 0.7-0.8). The earlier POOLED variant (concatenate all samples,
then count unique/total) puts Gen-0 Distinct-1 near ~0.19 and does NOT match
Tables 3-4; it is retained here only as ``distinct_n_pooled`` /
``type_token_ratio_pooled`` for comparison. If a metric panel ever shows Gen-0
Distinct-1 ~0.19, the pooled definition crept back in.

For unigrams, TTR is identical to Distinct-1 by construction (unique tokens /
total tokens). The two columns will match; report only one in the paper.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Tuple

try:
    import torch  # only needed for perplexity
except ImportError:  # pragma: no cover
    torch = None


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #
def tokenize(text: str) -> List[str]:
    """Word-level tokenizer used for all diversity metrics. Lowercased, punctuation stripped."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in text.split() if len(t) > 0]


def get_ngrams(tokens: List[str], n: int) -> List[Tuple]:
    if n <= 1:
        return [(t,) for t in tokens]
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# --------------------------------------------------------------------------- #
# Tier 1: diversity metrics (early warning) — PER-SAMPLE AVERAGED
# --------------------------------------------------------------------------- #
def distinct_n(texts: List[str], n: int) -> float:
    """Distinct-n = mean over samples of (|unique n-grams| / |n-grams|)."""
    vals = []
    for text in texts:
        grams = get_ngrams(tokenize(text), n)
        if grams:
            vals.append(len(set(grams)) / len(grams))
    return sum(vals) / len(vals) if vals else 0.0


def type_token_ratio(texts: List[str]) -> float:
    """TTR = mean over samples of (|unique tokens| / |tokens|)."""
    vals = []
    for text in texts:
        toks = tokenize(text)
        if toks:
            vals.append(len(set(toks)) / len(toks))
    return sum(vals) / len(vals) if vals else 0.0


# --------------------------------------------------------------------------- #
# Pooled variants — kept for reference/debugging only (NOT paper-scale)
# --------------------------------------------------------------------------- #
def distinct_n_pooled(texts: List[str], n: int) -> float:
    """Pooled Distinct-n across the whole sample set. Gen-0 ~0.19; do not use
    for the paper tables. Retained to document the original definition."""
    all_ngrams = []
    for text in texts:
        all_ngrams.extend(get_ngrams(tokenize(text), n))
    return len(set(all_ngrams)) / len(all_ngrams) if all_ngrams else 0.0


def type_token_ratio_pooled(texts: List[str]) -> float:
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenize(text))
    return len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0


def vocabulary_size(texts: List[str]) -> int:
    vocab = set()
    for text in texts:
        vocab.update(tokenize(text))
    return len(vocab)


def token_entropy(texts: List[str]) -> float:
    """Shannon entropy (bits) of the empirical token distribution."""
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenize(text))
    if not all_tokens:
        return 0.0
    counter = Counter(all_tokens)
    total = len(all_tokens)
    h = 0.0
    for c in counter.values():
        p = c / total
        h -= p * math.log2(p)
    return h


# --------------------------------------------------------------------------- #
# Tier 3: severity metrics
# --------------------------------------------------------------------------- #
def vocabulary_coverage(texts: List[str], reference_texts: List[str]) -> float:
    """Fraction of reference vocabulary appearing in generated samples."""
    gen_vocab = set()
    for t in texts:
        gen_vocab.update(tokenize(t))
    ref_vocab = set()
    for t in reference_texts:
        ref_vocab.update(tokenize(t))
    if not ref_vocab:
        return 0.0
    return len(gen_vocab & ref_vocab) / len(ref_vocab)


def repetition_rate(texts: List[str]) -> float:
    """Per-sample average of (1 - unique/total tokens) — the lexical-repetition
    proxy used in colab_notebook.ipynb for Table 9.

    NOTE: an alternative definition (fraction of tokens that immediately repeat
    the previous token) is available as ``repetition_rate_consecutive``. The two
    give different magnitudes; choose ONE for Table 9 and state it in the paper.
    """
    vals = []
    for text in texts:
        toks = tokenize(text)
        if toks:
            vals.append(1.0 - len(set(toks)) / len(toks))
    return sum(vals) / len(vals) if vals else 0.0


def repetition_rate_consecutive(texts: List[str]) -> float:
    """Fraction of tokens that are immediate repeats of the previous token
    (the 'the the process' degeneration pattern). Alternative to repetition_rate."""
    repeats = 0
    total = 0
    for text in texts:
        toks = tokenize(text)
        for i in range(1, len(toks)):
            total += 1
            if toks[i] == toks[i - 1]:
                repeats += 1
    return (repeats / total) if total else 0.0


# --------------------------------------------------------------------------- #
# Tier 2: perplexity (confirmation)
# --------------------------------------------------------------------------- #
def compute_perplexity(model, tokenizer, texts: List[str], max_length: int = 512) -> float:
    """Corpus-level perplexity = exp(total_nll / total_tokens) on a fixed,
    held-out test set. The test set must never be used in training."""
    if torch is None:
        raise ImportError("compute_perplexity requires torch")
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            n = inputs["input_ids"].numel()
            total_loss += outputs.loss.item() * n
            total_tokens += n
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


# --------------------------------------------------------------------------- #
# Comparison baseline: Self-BLEU (Table 6 in the paper)
# --------------------------------------------------------------------------- #
def self_bleu(texts: List[str], max_n: int = 4) -> float:
    """Self-BLEU: each sample scored as hypothesis against all others as
    references; return the mean. Higher == lower diversity. O(n^2)."""
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Self-BLEU requires nltk: pip install nltk") from exc
    smooth = SmoothingFunction().method1
    weights = tuple(1.0 / max_n for _ in range(max_n))
    tokenized = [tokenize(t) for t in texts]
    scores = []
    for i, hyp in enumerate(tokenized):
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
        if not hyp or not refs:
            continue
        scores.append(sentence_bleu(refs, hyp, weights=weights, smoothing_function=smooth))
    return (sum(scores) / len(scores)) if scores else 0.0


# --------------------------------------------------------------------------- #
# Convenience: compute the full metric panel for one generation's output
# --------------------------------------------------------------------------- #
def compute_all_metrics(
    texts: List[str],
    model=None,
    tokenizer=None,
    test_texts: "List[str] | None" = None,
    reference_texts: "List[str] | None" = None,
    include_self_bleu: bool = False,
) -> Dict:
    """Returns every metric for one generation's worth of samples.

    Perplexity is computed only if model/tokenizer/test_texts are provided.
    Self-BLEU is computed only if include_self_bleu=True (O(n^2)).
    Vocabulary coverage is computed only if reference_texts is provided.
    """
    metrics = {
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "distinct_3": distinct_n(texts, 3),
        "ttr": type_token_ratio(texts),
        "vocab_size": vocabulary_size(texts),
        "entropy": token_entropy(texts),
        "repetition_rate": repetition_rate(texts),
        "num_samples": len(texts),
        "avg_length": (sum(len(tokenize(t)) for t in texts) / max(len(texts), 1)),
    }
    if reference_texts is not None:
        metrics["vocab_coverage"] = vocabulary_coverage(texts, reference_texts)
    if model is not None and tokenizer is not None and test_texts is not None:
        metrics["perplexity"] = compute_perplexity(model, tokenizer, test_texts)
    if include_self_bleu:
        metrics["self_bleu"] = self_bleu(texts)
    return metrics
