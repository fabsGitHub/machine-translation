import os
import yaml

DEFAULT_CONFIG = {
    "system": {
        "seed": 42,
        "float32_matmul_precision": "high",
    },
    "data": {
        "sample_rate": 0.1,  # PDF requires a random 10% sample for training
        "test_split": 0.2,  # PDF requires 20 percent test set
        "seed": 42,
        "max_word_len": 50,
        "max_char_len": 300,
        "max_vocab_size": 30000,
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "num_workers": 4,
        "pin_memory": True,
        "prefetch_factor": 4,
        "persistent_workers": True,
    },
    "training": {
        "epochs": 1,
        "batch_size_word": 128,
        "batch_size_char": 256,
        "grad_accum_steps": 1,
        "precision": "bfloat16",
        "use_8bit_adam": True,
    },
    "profiles": {
        "word": {
            "lr": 0.001,
            "dropout": 0.3,
            "emb_dim": 256,
            "hidden_dim": 256,
            "batch_size": 128,
        },
        "char": {
            "lr": 0.001,
            "dropout": 0.3,
            "emb_dim": 64,
            "hidden_dim": 512,
            "batch_size": 256,
        },
    },
}


def load_config(config_path="config/config.yaml"):
    """Loads operational thresholds and parameter profiles safely from disk with fallback defaults."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded and isinstance(loaded, dict):
                    # Perform deep dictionary merge so explicit configs retain hardware defaults
                    merged = DEFAULT_CONFIG.copy()
                    for k, v in loaded.items():
                        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
                            merged[k] = {**merged[k], **v}
                        else:
                            merged[k] = v
                    return merged
        except Exception:
            pass
    return DEFAULT_CONFIG