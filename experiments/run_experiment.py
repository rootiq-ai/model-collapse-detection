#!/usr/bin/env python3
"""
Model Collapse Experiment
Run recursive fine-tuning on LLaMA 3 and GPT-2.
Includes: multi-seed runs, mixed-ratio experiments, and a CPU --dry-run that
writes a synthetic results JSON (no GPU/models) so analyze.py and
threshold_sensitivity.py can be exercised end-to-end.

Aligned with colab_notebook.ipynb:
  * per-model learning rate (GPT-2 uses 2e-5, LLaMA QLoRA uses 2e-4)
  * plain Trainer + DataCollatorForLanguageModeling (stable across TRL versions)

USAGE
    python experiments/run_experiment.py --model gpt2 --scenario replace --multi-seed
    python experiments/run_experiment.py --model all --scenario both --self-bleu
    python experiments/run_experiment.py --dry-run          # CPU, synthetic JSON
"""

import os
import json
import math
import random
import argparse
import numpy as np
from typing import List, Dict

from config import CONFIG, TRAIN_CORPUS, TEST_CORPUS, GENERATION_PROMPTS
from metrics import compute_all_metrics, distinct_n, type_token_ratio, \
    token_entropy, vocabulary_size, repetition_rate, vocabulary_coverage, tokenize

# Set by --self-bleu CLI flag; defaults to False (O(n^2) is expensive)
COMPUTE_SELF_BLEU = False


def _lr_for(model_name: str) -> float:
    return CONFIG.get("learning_rate_by_model", {}).get(model_name, CONFIG["learning_rate"])


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Model loading (torch/transformers imported lazily)
# --------------------------------------------------------------------------- #
def load_llama3():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print("\nLoading LLaMA 3-8B with QLoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["models"]["llama3"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["models"]["llama3"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print(f"LLaMA 3 loaded. GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer


def load_gpt2():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("\nLoading GPT-2 Medium...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["models"]["gpt2"], torch_dtype=torch.float16,
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["models"]["gpt2"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"GPT-2 loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    return model, tokenizer


# --------------------------------------------------------------------------- #
# Generation / training
# --------------------------------------------------------------------------- #
def generate_samples(model, tokenizer, num_samples: int) -> List[str]:
    import torch
    model.eval()
    generated_texts = []
    for i in range(num_samples):
        prompt = GENERATION_PROMPTS[i % len(GENERATION_PROMPTS)]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=CONFIG["max_new_tokens"],
                temperature=CONFIG["temperature"],
                top_p=CONFIG["top_p"],
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
        if len(generated) > CONFIG["min_kept_chars"]:
            generated_texts.append(generated)
        if (i + 1) % 10 == 0:
            print(f"   Generated {i + 1}/{num_samples}")
    return generated_texts


def train_model(model, tokenizer, texts: List[str], gen: int, output_dir: str, model_name: str):
    import torch
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    from datasets import Dataset

    lr = _lr_for(model_name)
    print(f"   Training on {len(texts)} samples  [lr={lr:.0e} for {model_name}]")
    clean = [t for t in texts if t and t.strip()]
    dataset = Dataset.from_dict({"text": clean})
    dataset = dataset.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=CONFIG["max_seq_length"], padding=False),
        batched=True, remove_columns=["text"],
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/gen{gen}",
        num_train_epochs=CONFIG["num_train_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=lr,
        fp16=True,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
    )
    # Cast trainable fp16 params to fp32 for stable optimization.
    for p in model.parameters():
        if p.requires_grad and p.dtype == torch.float16:
            p.data = p.data.float()
    Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=collator).train()
    print("   Training complete.")
    return model


# --------------------------------------------------------------------------- #
# Experiment drivers
# --------------------------------------------------------------------------- #
def run_single_experiment(model_name: str, output_dir: str, seed: int, scenario: str = "replace"):
    """Run a single experiment with the given seed and scenario."""
    import gc
    import torch
    set_seed(seed)

    print(f"\n{'=' * 70}")
    print(f"{model_name.upper()} | Seed: {seed} | Scenario: {scenario}")
    print(f"{'=' * 70}")

    model, tokenizer = (load_llama3() if model_name == "llama3" else load_gpt2())
    num_gens = CONFIG["mixed_generations"] if scenario == "mixed" else CONFIG["num_generations"]
    all_metrics, all_texts = {}, {}

    print("\nGeneration 0: training on human corpus")
    model = train_model(model, tokenizer, TRAIN_CORPUS, 0, output_dir, model_name)
    gen_texts = generate_samples(model, tokenizer, CONFIG["samples_per_generation"])
    all_texts[0] = gen_texts
    all_metrics[0] = compute_all_metrics(gen_texts, model, tokenizer, TEST_CORPUS,
                                         reference_texts=TRAIN_CORPUS, include_self_bleu=COMPUTE_SELF_BLEU)
    print(f"   D-1={all_metrics[0]['distinct_1']:.3f}  PPL={all_metrics[0]['perplexity']:.1f}")

    for gen in range(1, num_gens):
        print(f"\nGeneration {gen} ({scenario})")
        if scenario == "replace":
            train_data = all_texts[gen - 1]
        else:  # mixed: synthetic from prev gen + original seed corpus
            n_synthetic = int(len(all_texts[gen - 1]) * CONFIG["mixed_synthetic_ratio"])
            synthetic = random.sample(all_texts[gen - 1], min(n_synthetic, len(all_texts[gen - 1])))
            train_data = synthetic + TRAIN_CORPUS
            random.shuffle(train_data)
        model = train_model(model, tokenizer, train_data, gen, output_dir, model_name)
        gen_texts = generate_samples(model, tokenizer, CONFIG["samples_per_generation"])
        all_texts[gen] = gen_texts
        all_metrics[gen] = compute_all_metrics(gen_texts, model, tokenizer, TEST_CORPUS,
                                               reference_texts=TRAIN_CORPUS, include_self_bleu=COMPUTE_SELF_BLEU)
        print(f"   D-1={all_metrics[gen]['distinct_1']:.3f}  PPL={all_metrics[gen]['perplexity']:.1f}")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_metrics


def bootstrap_ci(values: List[float], n_bootstrap: int = None, ci: float = 0.95) -> tuple:
    """Bootstrap confidence interval (mean, lower, upper)."""
    n_bootstrap = n_bootstrap or CONFIG["bootstrap_resamples"]
    values = np.array(values, dtype=float)
    boots = np.array([np.mean(np.random.choice(values, size=len(values), replace=True))
                      for _ in range(n_bootstrap)])
    lower = np.percentile(boots, (1 - ci) / 2 * 100)
    upper = np.percentile(boots, (1 + ci) / 2 * 100)
    return float(np.mean(values)), float(lower), float(upper)


def run_multi_seed_experiment(model_name: str, output_dir: str, scenario: str = "replace"):
    """Run the experiment with multiple seeds and aggregate with bootstrap CIs."""
    all_runs = [run_single_experiment(model_name, f"{output_dir}/seed{seed}", seed, scenario)
                for seed in CONFIG["seeds"]]
    aggregated = {}
    num_gens = len(all_runs[0])
    for gen in range(num_gens):
        gen_metrics = {}
        for metric in ["distinct_1", "distinct_2", "ttr", "entropy", "perplexity", "vocab_size"]:
            if metric not in all_runs[0][gen]:
                continue
            values = [run[gen][metric] for run in all_runs]
            mean, lo, hi = bootstrap_ci(values)
            gen_metrics[metric] = mean
            gen_metrics[f"{metric}_ci_low"] = lo
            gen_metrics[f"{metric}_ci_high"] = hi
        aggregated[gen] = gen_metrics
    return aggregated, all_runs


def print_results(metrics: Dict, model_name: str, with_ci: bool = False):
    baseline = metrics[0]
    print(f"\n{'=' * 80}\nRESULTS: {model_name.upper()}\n{'=' * 80}")
    print(f"\n{'Gen':<5} {'D-1':<12} {'PPL':<12} {'dD-1':<10} {'dPPL':<10}")
    print("-" * 55)
    for gen in sorted(metrics.keys()):
        m = metrics[gen]
        d1c = "--" if gen == 0 else f"{(m['distinct_1'] / baseline['distinct_1'] - 1) * 100:+.1f}%"
        pplc = "--" if gen == 0 else f"{(m['perplexity'] / baseline['perplexity'] - 1) * 100:+.1f}%"
        print(f"{gen:<5} {m['distinct_1']:<12.4f} {m['perplexity']:<12.1f} {d1c:<10} {pplc:<10}")


# --------------------------------------------------------------------------- #
# CPU dry-run: synthetic results JSON (no GPU, no models)
# --------------------------------------------------------------------------- #
def _synthetic_panels(scenario: str = "replace") -> Dict[int, Dict]:
    """Build a deliberately synthetic degrading metric panel for plumbing tests.
    NOT an experimental result. Diversity drops; a toy perplexity rises slower."""
    rng = random.Random(42)
    pool = []
    for t in TRAIN_CORPUS:
        pool.extend(tokenize(t))
    from collections import Counter
    pool = [w for w, _ in Counter(pool).most_common()]
    num_gens = CONFIG["mixed_generations"] if scenario == "mixed" else CONFIG["num_generations"]
    panels = {}
    for g in range(num_gens):
        keep = max(8, int(len(pool) * (0.9 ** g)))
        rep_bias = min(0.85, 0.22 + 0.05 * g)
        active = pool[:keep]
        texts = []
        for _ in range(CONFIG["samples_per_generation"]):
            words, last = [], rng.choice(active)
            for _ in range(rng.randint(40, 70)):
                if rng.random() < rep_bias:
                    words.append(last)
                else:
                    last = rng.choice(active)
                    words.append(last)
            texts.append(" ".join(words) + ".")
        panels[g] = {
            "distinct_1": distinct_n(texts, 1),
            "distinct_2": distinct_n(texts, 2),
            "distinct_3": distinct_n(texts, 3),
            "ttr": type_token_ratio(texts),
            "vocab_size": vocabulary_size(texts),
            "entropy": token_entropy(texts),
            "repetition_rate": repetition_rate(texts),
            "vocab_coverage": vocabulary_coverage(texts, TRAIN_CORPUS),
            "perplexity": 24.2 * (1.0 + 0.012 * g + 0.010 * g * g),  # toy curve
            "num_samples": len(texts),
            "synthetic": True,
        }
    return panels


def run_dry(output_dir: str):
    banner = "PIPELINE SELF-TEST -- NOT EXPERIMENTAL RESULTS (synthetic toy data)"
    print("\n" + "#" * 72 + f"\n# {banner}\n" + "#" * 72)
    results = {
        "gpt2_replace": _synthetic_panels("replace"),
        "gpt2_mixed": _synthetic_panels("mixed"),
        "llama3_replace": _synthetic_panels("replace"),
    }
    print_results(results["gpt2_replace"], "GPT-2 (synthetic self-test)")
    path = os.path.join(output_dir, "selftest_results.json")
    with open(path, "w") as f:
        json.dump({k: {str(g): v for g, v in m.items()} for k, m in results.items()}, f, indent=2)
    print(f"\nWrote {path}")
    print("Now exercise the analysis on it:")
    print(f"   python experiments/analyze.py --input {path}")
    print(f"   python experiments/threshold_sensitivity.py --input {path}")


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Model Collapse Experiment")
    parser.add_argument("--model", choices=["llama3", "gpt2", "all"], default="all")
    parser.add_argument("--scenario", choices=["replace", "mixed", "both"], default="both")
    parser.add_argument("--multi-seed", action="store_true", help="Run with multiple seeds (GPT-2)")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--self-bleu", action="store_true", help="Also compute Self-BLEU (Table 6; O(n^2))")
    parser.add_argument("--dry-run", action="store_true", help="CPU synthetic results JSON (no GPU/models)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.dry_run:
        run_dry(args.output)
        return

    global COMPUTE_SELF_BLEU
    COMPUTE_SELF_BLEU = bool(args.self_bleu)

    results = {}

    if args.model in ["llama3", "all"] and args.scenario in ["replace", "both"]:
        llama_replace = run_single_experiment("llama3", f"{args.output}/llama3_replace",
                                              CONFIG["seeds"][0], "replace")
        print_results(llama_replace, "LLaMA 3-8B (Replace)")
        results["llama3_replace"] = llama_replace

    if args.model in ["gpt2", "all"]:
        if args.scenario in ["replace", "both"]:
            if args.multi_seed:
                gpt2_replace, _ = run_multi_seed_experiment("gpt2", f"{args.output}/gpt2_replace", "replace")
                print_results(gpt2_replace, "GPT-2 (Replace, 3 seeds)", with_ci=True)
            else:
                gpt2_replace = run_single_experiment("gpt2", f"{args.output}/gpt2_replace",
                                                    CONFIG["seeds"][0], "replace")
                print_results(gpt2_replace, "GPT-2 (Replace)")
            results["gpt2_replace"] = gpt2_replace

        if args.scenario in ["mixed", "both"]:
            gpt2_mixed = run_single_experiment("gpt2", f"{args.output}/gpt2_mixed",
                                              CONFIG["seeds"][0], "mixed")
            print_results(gpt2_mixed, "GPT-2 (Mixed 30%)")
            results["gpt2_mixed"] = gpt2_mixed

    with open(f"{args.output}/all_results.json", "w") as f:
        json.dump({k: {str(g): v for g, v in m.items()} for k, m in results.items()}, f, indent=2)
    print(f"\nAll experiments complete. Results saved to {args.output}/")


if __name__ == "__main__":
    main()
