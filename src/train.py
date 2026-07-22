import argparse
from contextlib import nullcontext
import json
import os
import random
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler

from config import load_config
from dataset import PAD_IDX, get_dataloader
from embeddings import (
    generate_word2vec_embeddings,
    load_glove_embeddings_pair,
)
from models import Decoder, Encoder, Seq2Seq
from utils import set_seed

# Enable cuDNN benchmark autotuning for static/similar batch shapes
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

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
        if hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)

    def __iter__(self):
        initial_seed = random.randint(0, 1000000)
        random.seed(self.epoch)
        batches = list(self.batch_sampler)
        if self.shuffle:
            random.shuffle(batches)

        if len(batches) % self.num_replicas != 0:
            padding_size = self.num_replicas - (
                len(batches) % self.num_replicas
            )
            batches += batches[:padding_size]

        random.seed(initial_seed + self.rank)
        for i in range(self.rank, len(batches), self.num_replicas):
            yield batches[i]

    def __len__(self):
        import math

        return math.ceil(len(self.batch_sampler) / self.num_replicas)


def get_clean_state_dict(model):
    raw_model = model.module if hasattr(model, "module") else model
    state_dict = raw_model.state_dict()
    clean_dict = {
        k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
    }
    return clean_dict


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "y", "1")


def get_vram_breakdown(model, optimizer, device):
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
            "vram_peak_gb": 0.0,
        }

    raw_model = model.module if hasattr(model, "module") else model
    model_bytes = sum(
        p.numel() * p.element_size() for p in raw_model.parameters()
    )
    grad_bytes = sum(
        p.grad.numel() * p.grad.element_size()
        for p in raw_model.parameters()
        if p.grad is not None
    )

    opt_bytes = 0
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                opt_bytes += v.numel() * v.element_size()

    allocated_bytes = torch.cuda.memory_allocated(device)
    peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
    peak_reserved_bytes = torch.cuda.max_memory_reserved(device)

    model_mb = model_bytes / (1024**2)
    grad_mb = grad_bytes / (1024**2)
    opt_mb = opt_bytes / (1024**2)
    peak_allocated_mb = peak_allocated_bytes / (1024**2)
    peak_reserved_mb = peak_reserved_bytes / (1024**2)

    activations_mb = max(
        0.0, peak_allocated_mb - (model_mb + grad_mb + opt_mb)
    )
    gpu_name = torch.cuda.get_device_name(device)

    return {
        "gpu_name": gpu_name,
        "vram_model_mb": round(model_mb, 2),
        "vram_gradients_mb": round(grad_mb, 2),
        "vram_optimizer_mb": round(opt_mb, 2),
        "vram_activations_mb": round(activations_mb, 2),
        "vram_allocated_mb": round(allocated_bytes / (1024**2), 2),
        "vram_reserved_mb": round(peak_reserved_mb, 2),
        "vram_peak_mb": round(peak_allocated_mb, 2),
        "vram_peak_gb": round(peak_reserved_bytes / (1024**3), 3),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Seq2Seq NMT Training Interface"
    )
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument(
        "--rnn_type",
        type=str,
        default="LSTM",
        choices=["RNN", "LSTM", "GRU"],
    )
    parser.add_argument("--bidirectional", type=str2bool, default=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--tf_ratio", type=float, default=0.4)
    parser.add_argument(
        "--attention_type",
        type=str,
        default="none",
        choices=["none", "luong", "bahdanau"],
    )
    parser.add_argument(
        "--token_type", type=str, default="word", choices=["word", "char"]
    )
    parser.add_argument(
        "--embedding_source",
        type=str,
        default="scratch",
        choices=["scratch", "word2vec", "glove"],
    )
    parser.add_argument("--freeze_emb", type=str2bool, default=False)
    parser.add_argument("--src_lang", type=str, default="de")
    parser.add_argument("--trg_lang", type=str, default="en")
    parser.add_argument(
        "--resume",
        type=str2bool,
        default=True,
        help="Resume from existing checkpoint if present",
    )
    return parser.parse_args()


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    clip,
    device,
    tf_ratio,
    scaler=None,
    vram_stats_out=None,
    grad_accum_steps=1,
):
    model.train()
    epoch_loss_tensor = torch.zeros((), device=device)
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (src, trg) in enumerate(dataloader):
        src, trg = src.to(device, non_blocking=True), trg.to(
            device, non_blocking=True
        )

        if batch_idx == 1 and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # Pure FP32 Forward Pass (Bypasses FP16 Autocast for GP104 Pascal Architecture)
        output = model(src, trg, teacher_forcing_ratio=tf_ratio)
        vocab_size = output.shape[-1]

        # Reshaped CrossEntropyLoss (Avoids expensive transpose().contiguous() allocations)
        loss = criterion(
            output[:, 1:].reshape(-1, vocab_size), trg[:, 1:].reshape(-1)
        )
        loss = loss / grad_accum_steps
        loss.backward()

        if batch_idx == 1 and vram_stats_out is not None and device.type == "cuda":
            vram_stats_out.update(get_vram_breakdown(model, optimizer, device))

        if (batch_idx + 1) % grad_accum_steps == 0 or (
            batch_idx + 1
        ) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss_tensor += loss.detach() * grad_accum_steps

    total_loss = (epoch_loss_tensor / len(dataloader)).item()

    if dist.is_initialized() and dist.get_world_size() > 1:
        loss_tensor = torch.tensor(total_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        return loss_tensor.item() / dist.get_world_size()

    return total_loss


def evaluate_validation(model, dataloader, criterion, device):
    model.eval()
    epoch_loss_tensor = torch.zeros((), device=device)
    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device, non_blocking=True), trg.to(
                device, non_blocking=True
            )
            output = model(src, trg, teacher_forcing_ratio=0.0)
            vocab_size = output.shape[-1]
            loss = criterion(
                output[:, 1:].reshape(-1, vocab_size), trg[:, 1:].reshape(-1)
            )
            epoch_loss_tensor += loss.detach()

    total_loss = (epoch_loss_tensor / len(dataloader)).item()

    if dist.is_initialized() and dist.get_world_size() > 1:
        loss_tensor = torch.tensor(total_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        return loss_tensor.item() / dist.get_world_size()

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

    set_seed(42 + rank)

    if rank == 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    cfg_data = load_config()
    processed_dir = cfg_data.get("data", {}).get(
        "processed_dir", "data/processed"
    )

    train_csv = os.path.join(
        processed_dir, f"train_{args.src_lang}_{args.trg_lang}.csv"
    )
    val_csv = os.path.join(
        processed_dir, f"val_{args.src_lang}_{args.trg_lang}.csv"
    )

    raw_train_loader, src_vocab, trg_vocab = get_dataloader(
        train_csv,
        batch_size=args.batch_size,
        shuffle=True,
        src_lang=args.src_lang,
        trg_lang=args.trg_lang,
        token_type=args.token_type,
    )
    raw_val_loader, _, _ = get_dataloader(
        val_csv,
        batch_size=args.batch_size,
        shuffle=False,
        src_vocab=src_vocab,
        trg_vocab=trg_vocab,
        src_lang=args.src_lang,
        trg_lang=args.trg_lang,
        token_type=args.token_type,
    )

    if is_distributed:
        train_sampler = DistributedBatchSamplerWrapper(
            raw_train_loader.batch_sampler,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        val_sampler = DistributedBatchSamplerWrapper(
            raw_val_loader.batch_sampler,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

        train_loader = DataLoader(
            raw_train_loader.dataset,
            batch_sampler=train_sampler,
            collate_fn=raw_train_loader.collate_fn,
            num_workers=raw_train_loader.num_workers,
            pin_memory=raw_train_loader.pin_memory,
            persistent_workers=(raw_train_loader.num_workers > 0),
        )
        val_loader = DataLoader(
            raw_val_loader.dataset,
            batch_sampler=val_sampler,
            collate_fn=raw_val_loader.collate_fn,
            num_workers=raw_val_loader.num_workers,
            pin_memory=raw_val_loader.pin_memory,
            persistent_workers=(raw_val_loader.num_workers > 0),
        )
    else:
        train_loader = raw_train_loader
        val_loader = raw_val_loader
        train_sampler = None
        val_sampler = None

    pretrained_src_emb, pretrained_trg_emb = None, None
    silent_logging = rank > 0
    pair_prefix = f"{args.src_lang}_{args.trg_lang}"

    if args.embedding_source == "word2vec":
        pretrained_src_emb = generate_word2vec_embeddings(
            src_vocab,
            train_csv,
            args.src_lang,
            args.emb_dim,
            silent=silent_logging,
            pair_prefix=pair_prefix,
        )
        pretrained_trg_emb = generate_word2vec_embeddings(
            trg_vocab,
            train_csv,
            args.trg_lang,
            args.emb_dim,
            silent=silent_logging,
            pair_prefix=pair_prefix,
        )
    elif args.embedding_source == "glove":
        glove_path = os.path.join(ROOT_DIR, "data", "glove.6B.300d.txt")
        pretrained_src_emb, pretrained_trg_emb = load_glove_embeddings_pair(
            src_vocab, trg_vocab, glove_path, 300, silent=silent_logging
        )

    num_directions = 2 if args.bidirectional else 1

    src_vocab_size = (
        src_vocab.padded_size
        if hasattr(src_vocab, "padded_size")
        else len(src_vocab)
    )
    trg_vocab_size = (
        trg_vocab.padded_size
        if hasattr(trg_vocab, "padded_size")
        else len(trg_vocab)
    )

    encoder = Encoder(
        src_vocab_size,
        args.emb_dim,
        args.hidden_dim,
        2,
        args.dropout,
        args.rnn_type,
        args.bidirectional,
        pretrained_src_emb,
        args.freeze_emb,
        300 if args.embedding_source == "glove" else None,
    )
    decoder = Decoder(
        trg_vocab_size,
        args.emb_dim,
        args.hidden_dim * num_directions,
        args.hidden_dim,
        2,
        args.dropout,
        args.rnn_type,
        args.attention_type,
        pretrained_trg_emb,
        args.freeze_emb,
        300 if args.embedding_source == "glove" else None,
    )

    model = Seq2Seq(encoder, decoder, device).to(device)

    if is_distributed:
        if device.type == "cuda":
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )
        else:
            model = nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=True
            )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scaler = None  # Disabled AMP FP16 for Pascal hardware (GTX 1070)

    best_val_loss = float("inf")
    start_train_time = time.time()
    loss_history = {"train": [], "val": []}

    exp_tag = (
        args.experiment
        if f"_{args.rnn_type}" in args.experiment
        else f"{args.experiment}_{args.rnn_type}"
    )
    checkpoint_path = os.path.join(OUTPUT_DIR, f"best_model_{exp_tag}.pt")
    config_json_path = os.path.join(OUTPUT_DIR, f"best_config_{exp_tag}.json")
    start_epoch = 0

    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(checkpoint["model_state_dict"])

        if "best_val_loss" in checkpoint.get("config", {}):
            best_val_loss = checkpoint["config"]["best_val_loss"]
        if "loss_history" in checkpoint and isinstance(
            checkpoint["loss_history"], dict
        ):
            loss_history = checkpoint["loss_history"]
            start_epoch = len(loss_history.get("train", []))

    vram_stats = {}

    if start_epoch < args.epochs:
        for epoch in range(start_epoch, args.epochs):
            epoch_start_time = time.time()

            if is_distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)
                val_sampler.set_epoch(epoch)

            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                args.clip,
                device,
                args.tf_ratio,
                scaler=scaler,
                vram_stats_out=vram_stats,
                grad_accum_steps=args.grad_accum_steps,
            )
            val_loss = evaluate_validation(
                model, val_loader, criterion, device
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            epoch_duration = time.time() - epoch_start_time
            loss_history["train"].append(train_loss)
            loss_history["val"].append(val_loss)

            if rank == 0:
                mins, secs = divmod(int(epoch_duration), 60)
                time_fmt = (
                    f"{mins:02d}m {secs:02d}s"
                    if mins > 0
                    else f"{epoch_duration:.2f}s"
                )
                print(
                    f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {time_fmt}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if rank == 0:
                    config_dict = vars(args).copy()
                    config_dict.update(
                        {
                            "train_time": f"{time.time() - start_train_time:.1f}",
                            "best_val_loss": best_val_loss,
                            "val_loss": best_val_loss,
                            "epochs_trained": len(loss_history["train"]),
                            "loss_history": loss_history,
                        }
                    )
                    config_dict.update(vram_stats)

                    torch.save(
                        {
                            "config": config_dict,
                            "model_state_dict": get_clean_state_dict(model),
                            "src_vocab": src_vocab,
                            "trg_vocab": trg_vocab,
                            "loss_history": loss_history,
                        },
                        checkpoint_path,
                    )

                    with open(config_json_path, "w") as f:
                        json.dump(config_dict, f, indent=4)
            else:
                if rank == 0:
                    print(
                        f"🛑 Early stopping triggered: Loss did not improve from {best_val_loss:.4f}."
                    )
                break

    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()