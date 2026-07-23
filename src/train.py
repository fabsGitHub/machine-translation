import argparse
import json
import math
import os
import random
import time
import torch

# Enable TensorFloat-32 (TF32) for Ampere Tensor Cores immediately at startup
torch.set_float32_matmul_precision('high')

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Sampler
from utils import set_seed, setup_logging

from config import load_config
from dataset import PAD_IDX, get_dataloader
from embeddings import generate_word2vec_embeddings, load_glove_embeddings_pair
from models import Decoder, Encoder, Seq2Seq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "results")


class DistributedBatchSamplerWrapper(Sampler):
    def __init__(self, batch_sampler, num_replicas, rank, shuffle=True):
        self.batch_sampler = batch_sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.batch_sampler, 'set_epoch'):
            self.batch_sampler.set_epoch(epoch)

    def __iter__(self):
        rng = random.Random(self.epoch + 42 + self.rank)
        batches = list(self.batch_sampler)
        if self.shuffle:
            rng.shuffle(batches)
            
        if len(batches) % self.num_replicas != 0:
            padding_size = self.num_replicas - (len(batches) % self.num_replicas)
            batches += batches[:padding_size]
            
        for i in range(self.rank, len(batches), self.num_replicas):
            yield batches[i]

    def __len__(self):
        return math.ceil(len(self.batch_sampler) / self.num_replicas)


def unwrap_model(model):
    """Recursively unwraps DistributedDataParallel and torch.compile wrappers."""
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
    return model


def get_clean_state_dict(model):
    raw_model = unwrap_model(model)
    state_dict = raw_model.state_dict()
    clean_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return clean_dict


def str2bool(v):
    if isinstance(v, bool): 
        return v
    return v.lower() in ('yes', 'true', 't', 'y', '1')


def setup_hardware_precision(device, precision_arg="auto"):
    """
    Dynamically configures hardware precision (FP32, FP16 AMP, BF16 AMP) 
    and hardware features (TF32, torch.compile) based on CUDA Compute Capability.
    """
    if device.type != "cuda":
        return "fp32", torch.float32, None, False, False

    major_cap, minor_cap = torch.cuda.get_device_capability(device)
    supports_tf32 = False
    
    if major_cap >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        supports_tf32 = True

    if precision_arg == "auto":
        if major_cap >= 8 and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            chosen_precision = "bf16"
        elif major_cap >= 7:
            chosen_precision = "fp16"
        else:
            chosen_precision = "fp32"
    else:
        chosen_precision = precision_arg.lower()

    if chosen_precision == "fp16":
        autocast_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda")
    elif chosen_precision == "bf16":
        autocast_dtype = torch.bfloat16
        scaler = None
    else:
        chosen_precision = "fp32"
        autocast_dtype = torch.float32
        scaler = None

    supports_compile = (major_cap >= 7) and hasattr(torch, "compile")
    return chosen_precision, autocast_dtype, scaler, supports_compile, supports_tf32


def get_vram_breakdown(model, optimizer, device):
    """Calculates PyTorch CUDA tensor memory allocation breakdown."""
    if not torch.cuda.is_available() or device.type != "cuda":
        return {
            "gpu_name": "CPU",
            "vram_model_mb": 0.0,
            "vram_gradients_mb": 0.0,
            "vram_optimizer_mb": 0.0,
            "vram_activations_mb": 0.0,
            "vram_allocated_mb": 0.0,
            "vram_reserved_mb": 0.0,
            "vram_peak_mb": 0.0,
            "vram_peak_gb": 0.0
        }
    
    raw_model = unwrap_model(model)
    
    model_bytes = sum(p.numel() * p.element_size() for p in raw_model.parameters())
    grad_bytes = sum(p.grad.numel() * p.grad.element_size() for p in raw_model.parameters() if p.grad is not None)
    
    opt_bytes = 0
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                opt_bytes += v.numel() * v.element_size()
                
    allocated_bytes = torch.cuda.memory_allocated(device)
    peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
    peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
    
    model_mb = model_bytes / (1024 ** 2)
    grad_mb = grad_bytes / (1024 ** 2)
    opt_mb = opt_bytes / (1024 ** 2)
    peak_allocated_mb = peak_allocated_bytes / (1024 ** 2)
    peak_reserved_mb = peak_reserved_bytes / (1024 ** 2)
    
    activations_mb = max(0.0, peak_allocated_mb - (model_mb + grad_mb + opt_mb))
    gpu_name = torch.cuda.get_device_name(device)
    
    return {
        "gpu_name": gpu_name,
        "vram_model_mb": round(model_mb, 2),
        "vram_gradients_mb": round(grad_mb, 2),
        "vram_optimizer_mb": round(opt_mb, 2),
        "vram_activations_mb": round(activations_mb, 2),
        "vram_allocated_mb": round(allocated_bytes / (1024 ** 2), 2),
        "vram_reserved_mb": round(peak_reserved_mb, 2),
        "vram_peak_mb": round(peak_allocated_mb, 2),
        "vram_peak_gb": round(peak_reserved_bytes / (1024 ** 3), 3)
    }


def configure_param_groups(model, weight_decay=1e-4):
    """
    Separates parameters into 2D weight matrices (weight decay enabled)
    and 1D bias, norm, or embedding parameters (weight decay disabled).
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 1D tensors (biases, layer norms) or embedding lookup tables
        if param.ndim <= 1 or "embedding" in name.lower() or "emb" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]


def compute_scheduled_tf_ratio(epoch, total_epochs, initial_tf=1.0, final_tf=0.3, mode="linear"):
    """Calculates scheduled teacher forcing ratio per epoch."""
    if total_epochs <= 1:
        return initial_tf
    progress = epoch / max(1, total_epochs - 1)
    if mode == "exponential":
        return initial_tf * ((final_tf / initial_tf) ** progress)
    else:  # Linear decay
        return initial_tf - progress * (initial_tf - final_tf)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Seq2Seq NMT Training Interface")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--rnn_type", type=str, default="LSTM", choices=["RNN", "LSTM", "GRU"])
    parser.add_argument("--bidirectional", type=str2bool, default=True)
    parser.add_argument("--epochs", type=int, default=5, help="Total training epochs (default set to 5)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs without loss improvement)")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Minimum validation loss change to count as improvement")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--warmup_epochs", type=int, default=1, help="Warmup epochs for LR scheduler (1 epoch default for 5-epoch run)")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--tf_start", type=float, default=1.0, help="Initial teacher forcing ratio")
    parser.add_argument("--tf_end", type=float, default=0.3, help="Final teacher forcing ratio (0.3 tuned for ~5 epochs)")
    parser.add_argument("--tf_decay_mode", type=str, default="linear", choices=["linear", "exponential"])
    parser.add_argument("--attention_type", type=str, default="none", choices=["none", "luong", "bahdanau"])
    parser.add_argument("--token_type", type=str, default="word", choices=["word", "char"])
    parser.add_argument("--embedding_source", type=str, default="scratch", choices=["scratch", "word2vec", "glove"])
    parser.add_argument("--freeze_emb", type=str2bool, default=False)
    parser.add_argument("--src_lang", type=str, default="de")
    parser.add_argument("--trg_lang", type=str, default="en")
    parser.add_argument("--resume", type=str2bool, default=True, help="Resume from existing checkpoint if present")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, criterion, clip, device, tf_ratio, 
                scaler=None, autocast_dtype=torch.float32, vram_stats_out=None, grad_accum_steps=1):
    model.train()
    epoch_loss_tensor = torch.zeros((), device=device)
    optimizer.zero_grad(set_to_none=True)
    
    use_amp = (device.type == "cuda") and (autocast_dtype in (torch.float16, torch.bfloat16))
    
    for batch_idx, (src, trg) in enumerate(dataloader):
        src, trg = src.to(device, non_blocking=True), trg.to(device, non_blocking=True)
        
        if batch_idx == 1 and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                output = model(src, trg, teacher_forcing_ratio=tf_ratio)
                output_dim = output.shape[-1]
                output_flat = output[:, 1:].reshape(-1, output_dim)
                trg_flat = trg[:, 1:].reshape(-1)

                loss = criterion(output_flat, trg_flat)
                loss = loss / grad_accum_steps
                
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            if batch_idx == 1 and vram_stats_out is not None:
                with torch.no_grad():
                    vram_stats_out.update(get_vram_breakdown(model, optimizer, device))
                
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            output = model(src, trg, teacher_forcing_ratio=tf_ratio)
            output_dim = output.shape[-1]
            output_flat = output[:, 1:].reshape(-1, output_dim)
            trg_flat = trg[:, 1:].reshape(-1)

            loss = criterion(output_flat, trg_flat)
            loss = loss / grad_accum_steps
            loss.backward()
            
            if batch_idx == 1 and vram_stats_out is not None and device.type == "cuda":
                with torch.no_grad():
                    vram_stats_out.update(get_vram_breakdown(model, optimizer, device))
                
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
        epoch_loss_tensor += (loss.detach() * grad_accum_steps)
    
    total_loss = (epoch_loss_tensor / len(dataloader)).item()
    
    if dist.is_initialized() and dist.get_world_size() > 1:
        loss_tensor = torch.tensor(total_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        return (loss_tensor / dist.get_world_size()).item()
    
    return total_loss


def evaluate_validation(model, dataloader, criterion, device, autocast_dtype=torch.float32):
    model.eval()
    epoch_loss_tensor = torch.zeros((), device=device)
    use_amp = (device.type == "cuda") and (autocast_dtype in (torch.float16, torch.bfloat16))

    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device, non_blocking=True), trg.to(device, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    output = model(src, trg, teacher_forcing_ratio=0.0)
                    output_dim = output.shape[-1]
                    output_flat = output[:, 1:].reshape(-1, output_dim)
                    trg_flat = trg[:, 1:].reshape(-1)
                    loss = criterion(output_flat, trg_flat)
            else:
                output = model(src, trg, teacher_forcing_ratio=0.0)
                output_dim = output.shape[-1]
                output_flat = output[:, 1:].reshape(-1, output_dim)
                trg_flat = trg[:, 1:].reshape(-1)
                loss = criterion(output_flat, trg_flat)
            epoch_loss_tensor += loss.detach()
            
    total_loss = (epoch_loss_tensor / len(dataloader)).item()
    
    if dist.is_initialized() and dist.get_world_size() > 1:
        loss_tensor = torch.tensor(total_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        return (loss_tensor / dist.get_world_size()).item()
        
    return total_loss


def main():
    args = parse_args()
    
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1

    if is_distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    else:
        device = torch.device("cpu")

    chosen_precision, autocast_dtype, scaler, can_compile, supports_tf32 = setup_hardware_precision(device, args.precision)
        
    set_seed(42 + rank)
    
    if rank == 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    cfg_data = load_config()
    processed_dir = cfg_data.get("data", {}).get("processed_dir", "data/processed")
    
    train_csv = os.path.join(processed_dir, f"train_{args.src_lang}_{args.trg_lang}.csv")
    val_csv = os.path.join(processed_dir, f"val_{args.src_lang}_{args.trg_lang}.csv")

    if rank == 0:
        exp_tag = args.experiment if f"_{args.rnn_type}" in args.experiment else f"{args.experiment}_{args.rnn_type}"
        setup_logging(log_filename=f"train_{exp_tag}.log", log_dir=OUTPUT_DIR, rank=rank)
        print(f"📁 Resolving train split: {train_csv}")
        print(f"📁 Resolving val split:   {val_csv}")

    num_workers = 8
    if is_distributed:
        raw_train_loader, src_vocab, trg_vocab = get_dataloader(
            train_csv, batch_size=args.batch_size, shuffle=True, 
            src_lang=args.src_lang, trg_lang=args.trg_lang, token_type=args.token_type,
            num_workers=0
        )
        raw_val_loader, _, _ = get_dataloader(
            val_csv, batch_size=args.batch_size, shuffle=False, 
            src_vocab=src_vocab, trg_vocab=trg_vocab, 
            src_lang=args.src_lang, trg_lang=args.trg_lang, token_type=args.token_type,
            num_workers=0
        )
        
        train_sampler = DistributedBatchSamplerWrapper(
            raw_train_loader.batch_sampler, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedBatchSamplerWrapper(
            raw_val_loader.batch_sampler, num_replicas=world_size, rank=rank, shuffle=False
        )
        
        train_loader = DataLoader(
            raw_train_loader.dataset,
            batch_sampler=train_sampler,
            collate_fn=raw_train_loader.collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        val_loader = DataLoader(
            raw_val_loader.dataset,
            batch_sampler=val_sampler,
            collate_fn=raw_val_loader.collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
    else:
        train_sampler = None
        val_sampler = None
        train_loader, src_vocab, trg_vocab = get_dataloader(
            train_csv, batch_size=args.batch_size, shuffle=True, 
            src_lang=args.src_lang, trg_lang=args.trg_lang, token_type=args.token_type,
            num_workers=num_workers
        )
        val_loader, _, _ = get_dataloader(
            val_csv, batch_size=args.batch_size, shuffle=False, 
            src_vocab=src_vocab, trg_vocab=trg_vocab, 
            src_lang=args.src_lang, trg_lang=args.trg_lang, token_type=args.token_type,
            num_workers=num_workers
        )
    
    pretrained_src_emb, pretrained_trg_emb = None, None
    silent_logging = rank > 0
    pair_prefix = f"{args.src_lang}_{args.trg_lang}"
    
    if args.embedding_source == "word2vec":
        pretrained_src_emb = generate_word2vec_embeddings(
            src_vocab, train_csv, lang=args.src_lang, emb_dim=300, silent=silent_logging, pair_prefix=pair_prefix
        )
        pretrained_trg_emb = generate_word2vec_embeddings(
            trg_vocab, train_csv, lang=args.trg_lang, emb_dim=300, silent=silent_logging, pair_prefix=pair_prefix
        )
    elif args.embedding_source == "glove":
        glove_dir = os.path.join(ROOT_DIR, "data")
        pretrained_src_emb, pretrained_trg_emb = load_glove_embeddings_pair(
            src_vocab, trg_vocab, src_lang=args.src_lang, trg_lang=args.trg_lang, 
            emb_dim=300, glove_dir=glove_dir, silent=silent_logging
        )
        
    num_directions = 2 if args.bidirectional else 1

    src_vocab_size = src_vocab.padded_size if hasattr(src_vocab, 'padded_size') else len(src_vocab)
    trg_vocab_size = trg_vocab.padded_size if hasattr(trg_vocab, 'padded_size') else len(trg_vocab)

    pretrained_src_dim = pretrained_src_emb.shape[1] if pretrained_src_emb is not None else None
    pretrained_trg_dim = pretrained_trg_emb.shape[1] if pretrained_trg_emb is not None else None

    encoder = Encoder(
        src_vocab_size, args.emb_dim, args.hidden_dim, 2, args.dropout, 
        args.rnn_type, args.bidirectional, pretrained_src_emb, args.freeze_emb, 
        custom_emb_dim=pretrained_src_dim
    )
    decoder = Decoder(
        trg_vocab_size, args.emb_dim, args.hidden_dim * num_directions, args.hidden_dim, 2, 
        args.dropout, args.rnn_type, args.attention_type, pretrained_trg_emb, args.freeze_emb, 
        custom_emb_dim=pretrained_trg_dim
    )
    
    model = Seq2Seq(encoder, decoder, device).to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = (total_params * 4) / (1024 ** 2)
        gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        effective_batch_per_gpu = args.batch_size * args.grad_accum_steps
        global_effective_batch = effective_batch_per_gpu * world_size

        print("\n" + "─" * 75)
        print(f"📐 [DYNAMIC MODEL & HARDWARE ANALYSIS]")
        print(f" ├─ Target Device:              {gpu_name}")
        print(f" ├─ Precision Strategy:         {chosen_precision.upper()} (TF32 Support: {supports_tf32})")
        print(f" ├─ Dynamic Compilation:        {'Supported' if can_compile else 'Disabled / Incompatible CC < 7.0'}")
        print(f" ├─ Experiment ID:              {args.experiment}")
        print(f" ├─ Architecture:               {args.rnn_type} ({'Bidirectional' if args.bidirectional else 'Unidirectional'})")
        print(f" ├─ Base Learning Rate:         {args.lr}")
        print(f" ├─ Micro-Batch Size (p/GPU):   {args.batch_size}")
        print(f" ├─ Grad Accumulation Steps:    {args.grad_accum_steps}")
        print(f" ├─ Global Effective Batch:     {global_effective_batch} sequence(s) across {world_size} rank(s)")
        print(f" ├─ Scheduled Teacher Forcing:  {args.tf_start:.2f} ➔ {args.tf_end:.2f} ({args.tf_decay_mode.capitalize()} Decay)")
        print(f" ├─ Early Stopping Patience:    {args.patience} epochs (Min Delta: {args.min_delta})")
        print(f" ├─ Total Trainable Parameters: {total_params:,}")
        print(f" └─ Parameter Weights (FP32):   {model_size_mb:.2f} MB")
        print("─" * 75 + "\n")
    
    best_val_loss = float("inf")
    patience_counter = 0
    start_train_time = time.time()
    loss_history = {"train": [], "val": [], "lr": [], "tf_ratio": []}
    
    exp_tag = args.experiment if f"_{args.rnn_type}" in args.experiment else f"{args.experiment}_{args.rnn_type}"
    checkpoint_path = os.path.join(OUTPUT_DIR, f"best_model_{exp_tag}.pt")
    config_json_path = os.path.join(OUTPUT_DIR, f"best_config_{exp_tag}.json")
    start_epoch = 0

    if can_compile:
        try:
            model.encoder = torch.compile(model.encoder, mode="default")
            if rank == 0:
                print("⚡ Compiled Encoder Graph with TorchInductor.")
        except Exception as e:
            if rank == 0:
                print(f"⚠️ torch.compile skipped: {e}")

    if is_distributed:
        if device.type == "cuda":
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
            )
        else:
            model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        
    # Configure parameter groups with filtered weight decay
    param_groups = configure_param_groups(model, weight_decay=args.weight_decay)
    
    if device.type == "cuda":
        optimizer = optim.AdamW(param_groups, lr=args.lr, fused=True)
        if rank == 0:
            print("⚡ Using Native PyTorch Fused AdamW Optimizer with Weight Decay filtering.")
    else:
        optimizer = optim.AdamW(param_groups, lr=args.lr)

    # Dynamic Warmup + Cosine Annealing LR Scheduler suited for short 5-epoch runs
    warmup_epochs = min(args.warmup_epochs, max(1, args.epochs // 4))
    cosine_epochs = max(1, args.epochs - warmup_epochs)
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    if args.resume and os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"🔄 Resuming model weights from existing checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        
        target_state = model.state_dict()
        adapted_state_dict = {}
        for k, v in clean_state_dict.items():
            if k in target_state:
                adapted_state_dict[k] = v
            elif k.replace("encoder.", "encoder._orig_mod.") in target_state:
                adapted_state_dict[k.replace("encoder.", "encoder._orig_mod.")] = v
            else:
                adapted_state_dict[k] = v

        model.load_state_dict(adapted_state_dict)
        
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                if rank == 0:
                    print("⚠️ Could not load optimizer state; maintaining fresh optimizer state.")
                    
        if 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception:
                pass

        if 'best_val_loss' in checkpoint.get('config', {}):
            best_val_loss = checkpoint['config']['best_val_loss']
        if 'loss_history' in checkpoint and isinstance(checkpoint['loss_history'], dict):
            loss_history = checkpoint['loss_history']
            start_epoch = len(loss_history.get("train", []))

    vram_stats = {}
    
    if start_epoch >= args.epochs:
        if rank == 0:
            print(f"📦 Checkpoint already fully trained ({start_epoch}/{args.epochs} epochs). Skipping epoch loop.")
    else:
        for epoch in range(start_epoch, args.epochs):
            epoch_start_time = time.time()
            
            # Dynamic Teacher Forcing Schedule Calculation
            current_tf_ratio = compute_scheduled_tf_ratio(
                epoch, args.epochs, initial_tf=args.tf_start, final_tf=args.tf_end, mode=args.tf_decay_mode
            )
            current_lr = optimizer.param_groups[0]['lr']

            if is_distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)
                val_sampler.set_epoch(epoch)
                
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, args.clip, device, current_tf_ratio, 
                scaler=scaler, autocast_dtype=autocast_dtype, vram_stats_out=vram_stats,
                grad_accum_steps=args.grad_accum_steps
            )
            val_loss = evaluate_validation(
                model, val_loader, criterion, device, autocast_dtype=autocast_dtype
            )
            
            # Step the learning rate scheduler at epoch boundaries
            scheduler.step()

            epoch_duration = time.time() - epoch_start_time
            loss_history["train"].append(train_loss)
            loss_history["val"].append(val_loss)
            loss_history["lr"].append(current_lr)
            loss_history["tf_ratio"].append(current_tf_ratio)
            
            if rank == 0 and epoch == start_epoch and vram_stats:
                print("─" * 75)
                print(f"📊 [PROFILED VRAM TENSOR MEMORY BREAKDOWN - {vram_stats.get('gpu_name', 'CUDA')}]")
                print(f" ├─ Model Weights VRAM:          {vram_stats.get('vram_model_mb', 0.0):>8.2f} MB")
                print(f" ├─ Gradients VRAM (p.grad):      {vram_stats.get('vram_gradients_mb', 0.0):>8.2f} MB")
                print(f" ├─ Optimizer State VRAM (Adam):  {vram_stats.get('vram_optimizer_mb', 0.0):>8.2f} MB")
                print(f" ├─ Dynamic Activations & Logits: {vram_stats.get('vram_activations_mb', 0.0):>8.2f} MB")
                print(f" ├─ Total Active Allocations:    {vram_stats.get('vram_allocated_mb', 0.0):>8.2f} MB")
                print(f" ├─ Reserved Memory Pool:         {vram_stats.get('vram_reserved_mb', 0.0):>8.2f} MB")
                print(f" └─ Peak Measured VRAM Footprint: {vram_stats.get('vram_peak_mb', 0.0):>8.2f} MB ({vram_stats.get('vram_peak_gb', 0.0):.3f} GB)")
                print("─" * 75 + "\n")

            if rank == 0:
                mins, secs = divmod(int(epoch_duration), 60)
                time_fmt = f"{mins:02d}m {secs:02d}s" if mins > 0 else f"{epoch_duration:.2f}s"
                print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f} | TF: {current_tf_ratio:.2f} | Time: {time_fmt}")
                
            # Early stopping check with patience and min_delta tolerance
            if val_loss < (best_val_loss - args.min_delta):
                best_val_loss = val_loss
                patience_counter = 0
                if rank == 0:
                    config_dict = vars(args).copy()
                    config_dict.update({
                        "train_time": round(time.time() - start_train_time, 1), 
                        "best_val_loss": best_val_loss, 
                        "val_loss": best_val_loss,
                        "epochs_trained": len(loss_history["train"]),
                        "loss_history": loss_history,
                        "hardware_precision": chosen_precision
                    })
                    config_dict.update(vram_stats)
                    
                    torch.save({
                        'config': config_dict, 
                        'model_state_dict': get_clean_state_dict(model), 
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'src_vocab': src_vocab, 
                        'trg_vocab': trg_vocab,
                        'loss_history': loss_history
                    }, checkpoint_path)
                    
                    with open(config_json_path, 'w') as f:
                        json.dump(config_dict, f, indent=4)
            else:
                patience_counter += 1
                if rank == 0:
                    print(f"⚠️ Validation loss did not improve ({val_loss:.4f} >= {best_val_loss:.4f}). Patience: {patience_counter}/{args.patience}")
                    try:
                        if os.path.exists(config_json_path):
                            with open(config_json_path, 'r') as f:
                                c_data = json.load(f)
                            c_data["loss_history"] = loss_history
                            c_data.update(vram_stats)
                            with open(config_json_path, 'w') as f:
                                json.dump(c_data, f, indent=4)
                    except Exception:
                        pass
                        
                if patience_counter >= args.patience:
                    if rank == 0:
                        print(f"🛑 Early stopping triggered after {patience_counter} epoch(s) without improvement.")
                    break

    if rank == 0:
        if os.path.exists(checkpoint_path) and not args.experiment.startswith("TUNE_"):
            try:
                import re
                import subprocess
                import sys
                
                evaluate_script = os.path.join(SCRIPT_DIR, "evaluate.py")
                if os.path.exists(evaluate_script):
                    print("\n⌛ Automated Backfill: Executing evaluation metrics extraction (BLEU & METEOR)...")
                    cmd = [sys.executable, evaluate_script, "evaluate", "--checkpoint", checkpoint_path]
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    
                    bleu_match = re.search(r"BLEU:\s*([\d\.]+)", result.stdout)
                    meteor_match = re.search(r"METEOR:\s*([\d\.]+)", result.stdout)
                    
                    bleu_score = float(bleu_match.group(1)) if bleu_match else None
                    meteor_score = float(meteor_match.group(1)) if meteor_match else None
                    
                    if bleu_score is not None:
                        if os.path.exists(config_json_path):
                            with open(config_json_path, 'r') as f:
                                c_data = json.load(f)
                            
                            c_data["bleu"] = bleu_score
                            c_data["Target Metric (BLEU)"] = bleu_score
                            c_data["bleu_score"] = bleu_score
                            c_data["overall_corpus_bleu"] = bleu_score
                            
                            if meteor_score is not None:
                                c_data["meteor"] = meteor_score
                                c_data["mean_meteor"] = meteor_score
                                
                            with open(config_json_path, 'w') as f:
                                json.dump(c_data, f, indent=4)
                                
                            print(f"✅ Backfill Successful: Saved BLEU={bleu_score} inside local JSON ledger.")
            except Exception as e:
                print(f"⚠️ Automated metrics backfill skipped: {e}")

    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()