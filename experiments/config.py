"""Configuration for Model Collapse Experiments.

Aligned with colab_notebook.ipynb. Two settings differ from the original repo
config and matter for reproducing the paper's ordering claim:

  * num_generations = 8 (was 6): a gentler GPT-2 fine-tune collapses more
    slowly, so a longer horizon keeps detection inside the window.
  * learning_rate_by_model: GPT-2 is fully fine-tuned and collapses in ONE
    generation at 2e-4, leaving no resolution to measure a lead time. 2e-5
    makes it degrade gradually (similar pace to LLaMA's QLoRA updates). This is
    not a thumb on the scale toward +2 — it just lets the experiment report
    whatever ordering is actually present.
"""

CONFIG = {
    "models": {
        "llama3": "meta-llama/Meta-Llama-3-8B",
        "gpt2": "gpt2-medium",
    },
    "num_generations": 8,
    "samples_per_generation": 50,
    "max_new_tokens": 150,
    "num_train_epochs": 1,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,

    # Default / fallback LR, plus per-model overrides (see module docstring).
    "learning_rate": 2e-4,
    "learning_rate_by_model": {
        "llama3": 2e-4,   # QLoRA: gentle low-rank updates
        "gpt2": 2e-5,     # full fine-tune: ~10x lower
    },

    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "temperature": 0.8,
    "top_p": 0.9,
    "max_seq_length": 512,
    "min_kept_chars": 50,

    # Detection
    "detection_threshold": 0.10,            # primary Tier-1/Tier-2 threshold
    "thresholds": [0.05, 0.10, 0.15, 0.20],  # Table 7 sweep
    "bootstrap_resamples": 1000,

    # Multi-seed for statistical rigor (GPT-2)
    "seeds": [42, 123, 456],

    # Mixed scenario settings
    "mixed_synthetic_ratio": 0.30,          # 30% synthetic, 70% real
    "mixed_generations": 9,                 # more generations for slower collapse
}

TRAIN_CORPUS = [
    "Photosynthesis is the biological process by which plants, algae, and certain bacteria convert light energy into chemical energy stored in glucose molecules. This remarkable transformation occurs primarily in chloroplasts, organelles containing chlorophyll pigments that absorb specific wavelengths of light. The process involves two stages: light-dependent reactions in thylakoid membranes and light-independent reactions in the stroma.",
    "The Renaissance period, spanning roughly from the 14th to 17th centuries, marked a profound cultural rebirth in European civilization. Originating in Florence, Italy, this movement emphasized humanism, scientific inquiry, and artistic innovation. Notable figures including Leonardo da Vinci, Michelangelo, and Raphael revolutionized painting techniques through perspective, anatomical accuracy, and emotional expression.",
    "Quantum mechanics fundamentally transformed our understanding of subatomic phenomena during the early twentieth century. The wave-particle duality proposed by Louis de Broglie suggested that electrons exhibit both particle and wave characteristics. Heisenberg's uncertainty principle established fundamental limits on simultaneously measuring position and momentum with arbitrary precision.",
    "The Amazon rainforest encompasses approximately 5.5 million square kilometers across nine South American nations. This biodiversity hotspot harbors roughly 10 percent of all species on Earth, including jaguars, pink river dolphins, and countless endemic plants. Deforestation threatens this ecosystem through agricultural expansion, illegal logging, and infrastructure development.",
    "Neural networks represent computational architectures loosely inspired by biological neurons. These systems learn patterns through iterative weight adjustments during training. Deep learning architectures stack multiple layers, enabling hierarchical feature extraction. Convolutional networks excel at image recognition while recurrent architectures process sequential data effectively.",
    "The Industrial Revolution transformed manufacturing processes beginning in 18th century Britain. Steam engines replaced water wheels, enabling factories to operate independently of river locations. Textile production mechanized rapidly through inventions like the spinning jenny and power loom. Urbanization accelerated as workers migrated from rural agricultural communities to industrial centers.",
    "Black holes represent regions where gravitational forces prevent anything, including electromagnetic radiation, from escaping. Stellar black holes form when massive stars exhaust nuclear fuel and collapse. Supermassive black holes inhabit galactic centers, containing millions to billions of solar masses. Event horizons mark boundaries beyond which escape becomes physically impossible.",
    "The human immune system comprises innate and adaptive components working synergistically against pathogens. Innate immunity provides immediate, nonspecific defense through physical barriers and phagocytic cells. Adaptive immunity develops specific responses through lymphocytes that recognize particular antigens. Immunological memory enables rapid secondary responses upon pathogen reencounter.",
    "Climate change results from anthropogenic greenhouse gas emissions altering atmospheric composition. Carbon dioxide concentrations have increased dramatically since industrialization began. Rising global temperatures affect precipitation patterns, sea levels, and ecosystem distributions. Mitigation strategies include renewable energy adoption, improved efficiency, and carbon capture technologies.",
    "Byzantine architecture synthesized Roman engineering traditions with Eastern artistic influences following Constantinople's establishment. Hagia Sophia exemplifies this style through its massive dome, pendentive supports, and elaborate mosaic decorations. Churches featured centralized plans with multiple domes creating complex interior spaces. Gold backgrounds in mosaics symbolized divine light and heavenly realms.",
]

TEST_CORPUS = [
    "The theory of evolution by natural selection, proposed by Charles Darwin, explains how species change over generations. Organisms with advantageous traits survive and reproduce more successfully, passing these traits to offspring. Genetic variation arises through mutations, recombination, and genetic drift. Fossil records and DNA evidence support common ancestry among all living organisms.",
    "Artificial intelligence encompasses machine learning algorithms designed to perform tasks requiring human-like cognition. Supervised learning trains models on labeled datasets to make predictions. Unsupervised learning discovers hidden patterns without explicit guidance. Reinforcement learning optimizes decision-making through reward signals in dynamic environments.",
    "The solar system formed approximately 4.6 billion years ago from a giant molecular cloud. Gravitational collapse created the Sun at the center, while remaining material formed planets, moons, and asteroids. Inner rocky planets differ significantly from outer gas giants in composition and size. Ongoing exploration reveals complex geological and atmospheric processes throughout the system.",
    "The French Revolution began in 1789 and fundamentally transformed European political structures. Enlightenment ideals of liberty, equality, and fraternity inspired revolutionary action against monarchical authority. The Declaration of the Rights of Man established principles of citizenship and individual rights. Subsequent turmoil led to the rise of Napoleon Bonaparte and decades of continental warfare.",
    "Cellular respiration converts glucose and oxygen into usable energy for living organisms. The process occurs in three stages: glycolysis in the cytoplasm, the citric acid cycle in mitochondria, and oxidative phosphorylation across the inner mitochondrial membrane. ATP molecules produced during respiration power virtually all cellular activities. Anaerobic alternatives exist for organisms in oxygen-poor environments.",
]

GENERATION_PROMPTS = [
    "Write an informative paragraph about science and nature:",
    "Explain an interesting concept in detail:",
    "Describe a fascinating topic:",
    "Write about an important subject:",
    "Discuss an interesting phenomenon:",
]
