"""Metrics for Model Collapse Detection"""

import re
import math
from collections import Counter
from typing import List, Dict, Tuple
import torch


def tokenize(text: str) -> List[str]:
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return [t for t in text.split() if len(t) > 0]


def get_ngrams(tokens: List[str], n: int) -> List[Tuple]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: List[str], n: int) -> float:
    all_ngrams = []
    for text in texts:
        tokens = tokenize(text)
        all_ngrams.extend(get_ngrams(tokens, n))
    if len(all_ngrams) == 0:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def type_token_ratio(texts: List[str]) -> float:
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenize(text))
    if len(all_tokens) == 0:
        return 0.0
    return len(set(all_tokens)) / len(all_tokens)


def vocabulary_size(texts: List[str]) -> int:
    all_tokens = set()
    for text in texts:
        all_tokens.update(tokenize(text))
    return len(all_tokens)


def token_entropy(texts: List[str]) -> float:
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenize(text))
    if len(all_tokens) == 0:
        return 0.0
    counter = Counter(all_tokens)
    total = len(all_tokens)
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def compute_perplexity(model, tokenizer, texts: List[str]) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            num_tokens = inputs["input_ids"].numel()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
    return math.exp(total_loss / total_tokens)


def compute_all_metrics(texts: List[str], model=None, tokenizer=None, test_texts=None) -> Dict:
    metrics = {
        'distinct_1': distinct_n(texts, 1),
        'distinct_2': distinct_n(texts, 2),
        'distinct_3': distinct_n(texts, 3),
        'ttr': type_token_ratio(texts),
        'vocab_size': vocabulary_size(texts),
        'entropy': token_entropy(texts),
        'num_samples': len(texts),
        'avg_length': sum(len(tokenize(t)) for t in texts) / max(len(texts), 1)
    }
    if model is not None and tokenizer is not None and test_texts is not None:
        metrics['perplexity'] = compute_perplexity(model, tokenizer, test_texts)
    return metrics
