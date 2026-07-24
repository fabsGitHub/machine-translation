"""
Standalone throughput/VRAM microbenchmark for the NMT training step, isolated from
run_studies.py / train.py's full CLI (checkpointing, DDP launch, eval backfill) so
timing reflects only the actual train_epoch work under different configs.

Run from the src/ directory: ../.venv/bin/python ../bench.py <variant>
Variants: baseline, threads_scoped, workers2, workers4, workers6, noamp, compile, compile_off
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("variant", choices=[
    "baseline", "threads_scoped", "workers2", "workers4", "workers6",
    "noamp", "compile_on", "compile_off",
])
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()

# Import AFTER deciding whether to patch torch.set_num_threads, since dataset.py
# calls it unconditionally at module scope.
if args.variant == "threads_scoped":
    # Simulate the "fixed" behavior: don't let dataset.py's global call take effect main-process-side.
    import dataset as _dataset_mod  # noqa
    torch.set_num_threads(os.cpu_count() or 4)  # restore full threads for main process after import
else:
    import dataset as _dataset_mod  # noqa  (this triggers torch.set_num_threads(1) as-is)

from dataset import get_dataloader, PAD_IDX
from models import Encoder, Decoder, Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_workers_map = {"workers2": 2, "workers4": 4, "workers6": 6}
num_workers = num_workers_map.get(args.variant, 8)

train_csv = "data/processed/train_de_en.csv"
train_loader, src_vocab, trg_vocab = get_dataloader(
    train_csv, batch_size=args.batch_size, shuffle=True,
    src_lang="de", trg_lang="en", token_type="word",
    num_workers=num_workers,
)

emb_dim, hidden_dim = 256, 512
encoder = Encoder(len(src_vocab), emb_dim, hidden_dim, 2, 0.3, "LSTM", True)
decoder = Decoder(len(trg_vocab), emb_dim, hidden_dim * 2, hidden_dim, 2, 0.3, "LSTM", "none")
model = Seq2Seq(encoder, decoder, device).to(device)

use_compile = args.variant == "compile_on"
if use_compile:
    model = torch.compile(model, dynamic=True)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
use_amp = device.type == "cuda" and args.variant != "noamp"
scaler = torch.amp.GradScaler("cuda") if use_amp else None

print(f"=== variant={args.variant} device={device} num_workers={num_workers} "
      f"batch_size={args.batch_size} amp={use_amp} compile={use_compile} "
      f"torch_num_threads={torch.get_num_threads()} ===")

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

model.train()
data_iter = iter(train_loader)


def next_batch():
    global data_iter
    try:
        return next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        return next(data_iter)


# Warmup (compile trace, cudnn autotune, worker startup)
warmup_start = time.time()
for _ in range(args.warmup):
    src, trg = next_batch()
    src, trg = src.to(device, non_blocking=True), trg.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    if use_amp:
        with torch.amp.autocast(device_type=device.type):
            output = model(src, trg, teacher_forcing_ratio=0.5)
            output_dim = output.shape[-1]
            out = output[:, :-1].reshape(-1, output_dim) if output.shape[1] == trg.shape[1] else output.reshape(-1, output_dim)
            loss = criterion(out, trg[:, 1:].reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        output = model(src, trg, teacher_forcing_ratio=0.5)
        output_dim = output.shape[-1]
        out = output[:, :-1].reshape(-1, output_dim) if output.shape[1] == trg.shape[1] else output.reshape(-1, output_dim)
        loss = criterion(out, trg[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
if device.type == "cuda":
    torch.cuda.synchronize()
warmup_time = time.time() - warmup_start

# Timed steps
n_sequences = 0
start = time.time()
for _ in range(args.steps):
    src, trg = next_batch()
    n_sequences += src.size(0)
    src, trg = src.to(device, non_blocking=True), trg.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    if use_amp:
        with torch.amp.autocast(device_type=device.type):
            output = model(src, trg, teacher_forcing_ratio=0.5)
            output_dim = output.shape[-1]
            out = output[:, :-1].reshape(-1, output_dim) if output.shape[1] == trg.shape[1] else output.reshape(-1, output_dim)
            loss = criterion(out, trg[:, 1:].reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        output = model(src, trg, teacher_forcing_ratio=0.5)
        output_dim = output.shape[-1]
        out = output[:, :-1].reshape(-1, output_dim) if output.shape[1] == trg.shape[1] else output.reshape(-1, output_dim)
        loss = criterion(out, trg[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
if device.type == "cuda":
    torch.cuda.synchronize()
elapsed = time.time() - start

peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0

print(f"RESULT variant={args.variant} warmup_s={warmup_time:.2f} steps={args.steps} "
      f"elapsed_s={elapsed:.3f} sec_per_step={elapsed/args.steps:.4f} "
      f"sequences_per_sec={n_sequences/elapsed:.1f} peak_vram_mb={peak_vram_mb:.1f} "
      f"final_loss={loss.item():.4f}")
