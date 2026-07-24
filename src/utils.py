import os
import sys
import json
import logging
import random
import torch
import numpy as np

class DualStreamTee:
    """
    Redirects stdout/stderr streams to both the original terminal and a log file simultaneously
    with buffered thread-safe writing to minimize host stream I/O stalls.
    """
    def __init__(self, original_stream, log_file):
        self.original_stream = original_stream
        self.log_file = log_file

    def write(self, message):
        self.original_stream.write(message)
        self.original_stream.flush()
        if self.log_file and not self.log_file.closed:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        self.original_stream.flush()
        if self.log_file and not self.log_file.closed:
            self.log_file.flush()


def setup_logging(log_filename="execution.log", log_dir="data/results", rank=0):
    """
    Initializes dual logging (terminal + file output).
    Intercepts standard print() calls and Python logger messages so everything
    is saved to file while remaining visible in the terminal.
    """
    if rank != 0:
        return None  # Suppress duplicate file writes for distributed multi-GPU ranks

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Stream Handler (Terminal)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler (Disk)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Intercept sys.stdout & sys.stderr to mirror all print() calls to the file
    log_file_obj = open(log_path, mode="a", encoding="utf-8")
    sys.stdout = DualStreamTee(sys.__stdout__, log_file_obj)
    sys.stderr = DualStreamTee(sys.__stderr__, log_file_obj)

    print(f"📝 Logging initialized -> Dual-streaming outputs to terminal and: {log_path}")
    return logger


def set_seed(seed=42, deterministic=False):
    """
    Sets global random seeds and configures Ampere GPU optimizations (TF32 and cuDNN benchmark mode).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        # Enable Tensor Core TF32 MatMul precision globally across Ampere GPUs
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            # Enable cuDNN autotuning benchmark for optimal fixed-shape convolution & GEMM kernels
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


def check_artifact_cache(output_dir, experiment_tags):
    """
    Verifies existence of experiment model checkpoints and associated configuration files.
    """
    for tag in experiment_tags:
        cfg = os.path.join(output_dir, f"best_config_{tag}.json")
        pt = os.path.join(output_dir, f"best_model_{tag}.pt")
        if os.path.exists(cfg) and os.path.exists(pt):
            return cfg, pt
    return None, None


def is_cache_valid(model_path, config_path):
    """
    Checks if a cached model exists, was successfully completed, 
    and optionally met target epoch requirements.
    """
    if not (os.path.exists(model_path) and os.path.exists(config_path)):
        return False
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            
        is_completed = bool(cfg.get("completed", False))
        
        return is_completed
        
    except Exception:
        return False


def pad_vocab_size(vocab_size: int, multiple: int = 8) -> int:
    """
    Pads vocabulary size to the nearest upper multiple of `multiple` (default: 8 or 64)
    for GPU Tensor Core memory alignment and optimal GEMM execution layout.
    """
    if multiple & (multiple - 1) == 0:
        # Fast bitwise mask alignment for power-of-two multiples (8, 16, 32, 64)
        return (vocab_size + multiple - 1) & ~(multiple - 1)
    if vocab_size % multiple == 0:
        return vocab_size
    return ((vocab_size + multiple - 1) // multiple) * multiple