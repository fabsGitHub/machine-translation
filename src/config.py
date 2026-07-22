import os
import yaml

DEFAULT_CONFIG = {
    "system": {
        "seed": 42
    },
    "data": {
        "sample_rate": 0.01,
        "test_split": 0.1,
        "seed": 42,
        "max_word_len": 50,
        "max_char_len": 300,
        "raw_dir": "data/raw",
        "processed_dir": "data/processed"
    },
    "training": {
        "epochs": 1
    },
    "profiles": {
        "word": {
            "lr": 0.001,
            "dropout": 0.3,
            "emb_dim": 256,
            "hidden_dim": 512
        },
        "char": {
            "lr": 0.001,
            "dropout": 0.3,
            "emb_dim": 64,
            "hidden_dim": 256
        }
    }
}

def load_config(config_path="config/config.yaml"):
    """Loads operational thresholds and parameter profiles safely from disk with fallback defaults."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded and isinstance(loaded, dict):
                    return loaded
        except Exception:
            pass
    return DEFAULT_CONFIG