import os
import random
import numpy as np
import torch
from config import load_config

def pad_vocab_size(vocab_len: int, multiple: int = 16) -> int:
    """
    Pads vocabulary size to the nearest multiple of 16 to trigger
    NVIDIA Ampere Tensor Core XMMA GEMM kernels.
    """
    return ((vocab_len + multiple - 1) // multiple) * multiple

def set_seed(seed=42, deterministic=False):
    """
    Configures seed settings across Python, NumPy, and PyTorch backends.
    Allows cuDNN autotuning when deterministic is set to False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def check_artifact_cache(output_dir, candidate_tags):
    """Iterates through candidate tags and returns cached JSON path if available."""
    for tag in candidate_tags:
        c_json = os.path.join(output_dir, f"best_config_{tag}.json")
        c_pt = os.path.join(output_dir, f"best_model_{tag}.pt")
        if os.path.exists(c_json) and os.path.exists(c_pt):
            return c_json, tag
    return None, None

def is_cache_valid(model_path, config_path, target_epochs=1):
    """Unconditional existence check for checkpoint artifacts."""
    if os.path.exists(model_path) and os.path.exists(config_path):
        print(f"📦 [Cache Hit] Valid artifacts found for {os.path.basename(model_path)}. Skipping execution.")
        return True
    return False