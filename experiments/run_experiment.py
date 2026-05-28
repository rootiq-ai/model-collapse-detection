#!/usr/bin/env python3
"""
Model Collapse Experiment
Run recursive fine-tuning on LLaMA 3 and GPT-2
Includes: multi-seed runs and mixed-ratio experiments
"""

import os
import json
import random
import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict
from copy import deepcopy

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset

from config import CONFIG, TRAIN_CORPUS, TEST_CORPUS, GENERATION_PROMPTS
from metrics import compute_all_metrics

# Set by --self-bleu CLI flag; defaults to False (O(n^2) is expensive)
COMPUTE_SELF_BLEU = False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_llama3():
    print("\n🦙 Loading LLaMA 3-8B with QLoRA...")
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
    print(f"✅ LLaMA 3 loaded! Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer


def load_gpt2():
    print("\n🤖 Loading GPT-2 Medium...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["models"]["gpt2"],
        torch_dtype=torch.float16,
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["models"]["gpt2"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"✅ GPT-2 loaded! Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer


def generate_samples(model, tokenizer, num_samples: int) -> List[str]:
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
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = generated[len(prompt):].strip()
        if len(generated) > 50:
            generated_texts.append(generated)
        if (i + 1) % 10 == 0:
            print(f"   Generated {i + 1}/{num_samples}")
    return generated_texts


def train_model(model, tokenizer, texts: List[str], gen: int, output_dir: str):
    print(f"   🔄 Training on {len(texts)} samples...")
    dataset = Dataset.from_dict({"text": texts})
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/gen{gen}",
        num_train_epochs=CONFIG["num_train_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        fp16=True,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=512,
    )
    trainer.train()
    print(f"   ✅ Training complete!")
    return model


def run_single_experiment(model_name: str, output_dir: str, seed: int, scenario: str = "replace"):
    """Run single experiment with given seed and scenario."""
    set_seed(seed)
    
    print(f"\n{'='*70}")
    print(f"🚀 {model_name.upper()} | Seed: {seed} | Scenario: {scenario}")
    print(f"{'='*70}")
    
    # Load model
    if model_name == "llama3":
        model, tokenizer = load_llama3()
    else:
        model, tokenizer = load_gpt2()
    
    num_gens = CONFIG["mixed_generations"] if scenario == "mixed" else CONFIG["num_generations"]
    all_metrics = {}
    all_texts = {}
    
    # Generation 0
    print(f"\n📍 Generation 0: Training on human corpus")
    model = train_model(model, tokenizer, TRAIN_CORPUS, 0, output_dir)
    print(f"   Generating samples...")
    gen_texts = generate_samples(model, tokenizer, CONFIG["samples_per_generation"])
    all_texts[0] = gen_texts
    all_metrics[0] = compute_all_metrics(gen_texts, model, tokenizer, TEST_CORPUS, reference_texts=TRAIN_CORPUS, include_self_bleu=COMPUTE_SELF_BLEU)
    print(f"   📊 D-1={all_metrics[0]['distinct_1']:.3f}, PPL={all_metrics[0]['perplexity']:.1f}")
    
    # Subsequent generations
    for gen in range(1, num_gens):
        print(f"\n📍 Generation {gen}")
        
        if scenario == "replace":
            # 100% synthetic from previous generation
            train_data = all_texts[gen-1]
        else:  # mixed
            # 30% synthetic + 70% real
            n_synthetic = int(len(all_texts[gen-1]) * CONFIG["mixed_synthetic_ratio"])
            n_real = CONFIG["samples_per_generation"] - n_synthetic
            synthetic_samples = random.sample(all_texts[gen-1], min(n_synthetic, len(all_texts[gen-1])))
            real_samples = random.choices(TRAIN_CORPUS, k=n_real)
            train_data = synthetic_samples + real_samples
            random.shuffle(train_data)
        
        model = train_model(model, tokenizer, train_data, gen, output_dir)
        print(f"   Generating samples...")
        gen_texts = generate_samples(model, tokenizer, CONFIG["samples_per_generation"])
        all_texts[gen] = gen_texts
        all_metrics[gen] = compute_all_metrics(gen_texts, model, tokenizer, TEST_CORPUS, reference_texts=TRAIN_CORPUS, include_self_bleu=COMPUTE_SELF_BLEU)
        print(f"   📊 D-1={all_metrics[gen]['distinct_1']:.3f}, PPL={all_metrics[gen]['perplexity']:.1f}")
    
    return all_metrics


def bootstrap_ci(values: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> tuple:
    """Compute bootstrap confidence interval."""
    values = np.array(values)
    bootstrapped = np.array([np.mean(np.random.choice(values, size=len(values), replace=True)) 
                             for _ in range(n_bootstrap)])
    lower = np.percentile(bootstrapped, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrapped, (1 + ci) / 2 * 100)
    return np.mean(values), lower, upper


def run_multi_seed_experiment(model_name: str, output_dir: str, scenario: str = "replace"):
    """Run experiment with multiple seeds and compute confidence intervals."""
    all_runs = []
    
    for seed in CONFIG["seeds"]:
        metrics = run_single_experiment(model_name, f"{output_dir}/seed{seed}", seed, scenario)
        all_runs.append(metrics)
    
    # Aggregate results with confidence intervals
    aggregated = {}
    num_gens = len(all_runs[0])
    
    for gen in range(num_gens):
        gen_metrics = {}
        for metric in ['distinct_1', 'distinct_2', 'ttr', 'perplexity']:
            values = [run[gen][metric] for run in all_runs]
            mean, ci_low, ci_high = bootstrap_ci(values)
            gen_metrics[metric] = mean
            gen_metrics[f"{metric}_ci_low"] = ci_low
            gen_metrics[f"{metric}_ci_high"] = ci_high
        aggregated[gen] = gen_metrics
    
    return aggregated, all_runs


def print_results(metrics: Dict, model_name: str, with_ci: bool = False):
    baseline = metrics[0]
    print(f"\n{'='*80}")
    print(f"📊 RESULTS: {model_name.upper()}")
    print(f"{'='*80}")
    
    if with_ci:
        print(f"\n{'Gen':<5} {'D-1':<15} {'PPL':<15} {'ΔD-1':<12} {'ΔPPL':<12}")
        print("-"*60)
        for gen in sorted(metrics.keys()):
            m = metrics[gen]
            d1_str = f"{m['distinct_1']:.3f}±{(m['distinct_1_ci_high']-m['distinct_1_ci_low'])/2:.3f}"
            ppl_str = f"{m['perplexity']:.1f}±{(m['perplexity_ci_high']-m['perplexity_ci_low'])/2:.1f}"
            d1_change = ((m['distinct_1'] / baseline['distinct_1']) - 1) * 100 if gen > 0 else 0
            ppl_change = ((m['perplexity'] / baseline['perplexity']) - 1) * 100 if gen > 0 else 0
            d1_c = f"{d1_change:+.1f}%" if gen > 0 else "--"
            ppl_c = f"{ppl_change:+.1f}%" if gen > 0 else "--"
            print(f"{gen:<5} {d1_str:<15} {ppl_str:<15} {d1_c:<12} {ppl_c:<12}")
    else:
        print(f"\n{'Gen':<5} {'D-1':<10} {'PPL':<10} {'ΔD-1':<10} {'ΔPPL':<10}")
        print("-"*50)
        for gen in sorted(metrics.keys()):
            m = metrics[gen]
            d1_change = ((m['distinct_1'] / baseline['distinct_1']) - 1) * 100 if gen > 0 else 0
            ppl_change = ((m['perplexity'] / baseline['perplexity']) - 1) * 100 if gen > 0 else 0
            d1_str = f"{d1_change:+.1f}%" if gen > 0 else "--"
            ppl_str = f"{ppl_change:+.1f}%" if gen > 0 else "--"
            print(f"{gen:<5} {m['distinct_1']:<10.4f} {m['perplexity']:<10.1f} {d1_str:<10} {ppl_str:<10}")


def main():
    parser = argparse.ArgumentParser(description="Model Collapse Experiment")
    parser.add_argument("--model", choices=["llama3", "gpt2", "all"], default="all")
    parser.add_argument("--scenario", choices=["replace", "mixed", "both"], default="both")
    parser.add_argument("--multi-seed", action="store_true", help="Run with multiple seeds (GPT-2 only)")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--self-bleu", action="store_true", help="Also compute Self-BLEU (Table 6 comparison; O(n^2), slower)")
    args = parser.parse_args()
    
    global COMPUTE_SELF_BLEU
    COMPUTE_SELF_BLEU = bool(args.self_bleu)
    
    os.makedirs(args.output, exist_ok=True)
    
    results = {}
    
    # LLaMA 3 experiments (single seed due to compute)
    if args.model in ["llama3", "all"]:
        if args.scenario in ["replace", "both"]:
            set_seed(CONFIG["seeds"][0])
            llama_replace = run_single_experiment("llama3", f"{args.output}/llama3_replace", CONFIG["seeds"][0], "replace")
            print_results(llama_replace, "LLaMA 3-8B (Replace)")
            results["llama3_replace"] = llama_replace
    
    # GPT-2 experiments (multi-seed)
    if args.model in ["gpt2", "all"]:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        if args.scenario in ["replace", "both"]:
            if args.multi_seed:
                gpt2_replace, _ = run_multi_seed_experiment("gpt2", f"{args.output}/gpt2_replace", "replace")
                print_results(gpt2_replace, "GPT-2 (Replace, 3 seeds)", with_ci=True)
            else:
                set_seed(CONFIG["seeds"][0])
                gpt2_replace = run_single_experiment("gpt2", f"{args.output}/gpt2_replace", CONFIG["seeds"][0], "replace")
                print_results(gpt2_replace, "GPT-2 (Replace)")
            results["gpt2_replace"] = gpt2_replace
        
        if args.scenario in ["mixed", "both"]:
            gc.collect()
            torch.cuda.empty_cache()
            set_seed(CONFIG["seeds"][0])
            gpt2_mixed = run_single_experiment("gpt2", f"{args.output}/gpt2_mixed", CONFIG["seeds"][0], "mixed")
            print_results(gpt2_mixed, "GPT-2 (Mixed 30%)")
            results["gpt2_mixed"] = gpt2_mixed
    
    # Save all results
    with open(f"{args.output}/all_results.json", "w") as f:
        json.dump({k: {str(g): v for g, v in m.items()} for k, m in results.items()}, f, indent=2)
    
    print("\n🎉 ALL EXPERIMENTS COMPLETE!")
    print(f"Results saved to {args.output}/")


if __name__ == "__main__":
    main()
