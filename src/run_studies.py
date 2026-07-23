import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, wait
import glob
import itertools
import json
import os
import random
import re
import subprocess
import sys
import threading
import time

import matplotlib

matplotlib.use('Agg')
import pandas as pd
import torch
from config import load_config
from evaluate import generate_all_reports, visualize_attention
from utils import check_artifact_cache, is_cache_valid, set_seed, setup_logging

# Optimize PyTorch CUDA memory allocator to prevent fragmentation on 24GB VRAM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Enable TensorCore TF32 execution globally for Ampere GPUs
torch.set_float32_matmul_precision('high')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "results")
CONFIG_PATH = os.path.join(REPO_ROOT, "config", "config.yaml")

config = load_config(CONFIG_PATH)
set_seed(config.get('system', {}).get('seed', 42))

eval_lock = threading.Lock()


def get_batch_size(study, token_type):
    """Centralized handler for batch size configuration across studies and token levels."""
    config_batch = config.get('training', {}).get('batch_size')
    if config_batch is not None:
        return str(config_batch)

    return "256" if token_type == "char" else "128"


class AsyncEvaluationQueue:
    """Offloads evaluation and ledger synchronization to background execution threads."""

    def __init__(self, max_workers=2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []

    def submit_evaluation(self, experiment_id, rnn_type, token_type):
        """Queues evaluation in the background without holding a global lock during computation."""

        def _task():
            print(
                f"\n⚡ [Async Eval Started] -> {experiment_id} ({rnn_type})"
            )
            run_auto_evaluation(experiment_id, rnn_type)
            with eval_lock:
                sync_ledger_to_token_type(token_type)
            print(
                f"✅ [Async Eval Finished] -> {experiment_id} ({rnn_type})"
            )

        future = self.executor.submit(_task)
        self.futures.append(future)

    def sync_study(self):
        """Blocks until all queued evaluations for the current study complete."""
        if self.futures:
            print(
                "\n⏳ [Study Barrier] Waiting for background evaluation tasks"
                " to complete..."
            )
            wait(self.futures)
            self.futures.clear()
            print(
                "🎯 [Study Barrier Cleared] All study models evaluated"
                " successfully.\n"
            )

    def shutdown(self):
        self.executor.shutdown(wait=True)


def get_vocab_sizes(token_type="word"):
    """Dynamically retrieves vocabulary size from binary dataset cache or defaults to baseline."""
    processed_dir = os.path.join(REPO_ROOT, "data", "processed")
    cache_dir = os.path.join(processed_dir, ".matrix_cache")
    if os.path.exists(cache_dir):
        for fname in os.listdir(cache_dir):
            if fname.endswith(".pt") and token_type in fname:
                try:
                    payload = torch.load(
                        os.path.join(cache_dir, fname),
                        map_location="cpu",
                        weights_only=False,
                    )
                    if isinstance(payload, dict) and "src_vocab" in payload:
                        return len(payload["src_vocab"]), len(
                            payload["trg_vocab"]
                        )
                except Exception:
                    pass
    return (8192, 8192) if token_type == "word" else (256, 256)


def print_study_model_and_batch_info(
    study_name,
    exp_id,
    token_type,
    rnn_type,
    bidirectional,
    attention_type,
    emb_dim,
    hidden_dim,
    batch_size,
):
    """Analytically computes and outputs Model Size and Batch Size parameters."""
    src_vocab_len, trg_vocab_len = get_vocab_sizes(token_type)
    bidi_bool = str(bidirectional).lower() == "true"
    num_directions = 2 if bidi_bool else 1
    emb_d, hid_d = int(emb_dim), int(hidden_dim)
    gates = 4 if rnn_type == "LSTM" else (3 if rnn_type == "GRU" else 1)

    enc_emb = src_vocab_len * emb_d
    enc_l1 = gates * (
        (emb_d * hid_d + hid_d * hid_d + 2 * hid_d) * num_directions
    )
    enc_l2 = gates * (
        (hid_d * num_directions * hid_d + hid_d * hid_d + 2 * hid_d)
        * num_directions
    )
    enc_params = enc_emb + enc_l1 + enc_l2

    dec_emb = trg_vocab_len * emb_d
    enc_out_dim = hid_d * num_directions
    dec_rnn_in = emb_d + (enc_out_dim if attention_type != "none" else 0)
    dec_l1 = gates * (dec_rnn_in * hid_d + hid_d * hid_d + 2 * hid_d)
    dec_l2 = gates * (hid_d * hid_d + hid_d * hid_d + 2 * hid_d)
    dec_fc = hid_d * trg_vocab_len + trg_vocab_len

    attn_params = 0
    if attention_type == "luong":
        attn_params = (enc_out_dim * hid_d) + hid_d
    elif attention_type == "bahdanau":
        attn_params = (
            (hid_d * hid_d + hid_d) + (enc_out_dim * hid_d + hid_d) + hid_d
        )

    total_params = enc_params + dec_emb + dec_l1 + dec_l2 + dec_fc + attn_params
    model_size_mb = (total_params * 4) / (1024**2)

    seq_len = 256 if token_type == "char" else 64
    batch_num = int(batch_size)
    batch_memory_mb = (batch_num * seq_len * 8) / (1024**2)

    print("\n" + "─" * 75)
    print(f"📐 [DYNAMIC STUDY ANALYSIS] - {study_name} ({exp_id})")
    print(f" ├─ Tokenizer Mode:           {token_type.upper()}")
    print(
        f" ├─ Architecture:             {rnn_type} (BiDirect: {bidi_bool},"
        f" Attention: {attention_type})"
    )
    print(f" ├─ Dimensions:               Emb={emb_dim} | Hidden={hidden_dim}")
    print(f" ├─ Batch Size (N samples):   {batch_num} sequences / batch")
    print(
        f" ├─ Batch Shape Estimate:     [{batch_num}, {seq_len}]"
        f" ({batch_memory_mb:.4f} MB per batch tensor)"
    )
    print(f" ├─ Total Model Parameters:   ~{total_params:,} parameters")
    print(f" └─ Model Memory Footprint:   ~{model_size_mb:.2f} MB")
    print("─" * 75 + "\n")


def run_cmd(args_list):
    """Executes distributed PyTorch training sub-process via PyTorch DDP launcher."""
    if "--grad_accum_steps" not in args_list:
        args_list = ["--grad_accum_steps", "4"] + args_list

    i = 0
    kv = {}
    positional = []
    while i < len(args_list):
        item = args_list[i]
        if item.startswith("-"):
            if i + 1 < len(args_list) and not args_list[i + 1].startswith("-"):
                kv[item] = args_list[i + 1]
                i += 2
            else:
                kv[item] = None
                i += 1
        else:
            positional.append(item)
            i += 1

    cleaned_args = []
    for k, v in kv.items():
        cleaned_args.append(k)
        if v is not None:
            cleaned_args.append(v)
    args_list = cleaned_args + positional

    epochs = config.get("training", {}).get("epochs", 1)
    if "--epochs" in args_list:
        try:
            epochs = int(args_list[args_list.index("--epochs") + 1])
        except (ValueError, IndexError):
            pass

    nproc = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        os.path.join(SCRIPT_DIR, "train.py"),
    ] + args_list

    print(
        f"\n🚀 Launching DDP Execution Unit ({nproc} processes):"
        f" {' '.join(command)}"
    )

    start_time = time.time()
    subprocess.run(command, check=True)
    duration = time.time() - start_time
    print(
        f"⏱️ Done. Duration: {duration:.2f}s | Avg/Epoch:"
        f" {duration/max(1, epochs):.2f}s"
    )


def run_auto_evaluation(experiment_id, rnn_type):
    """Executes model evaluation with GPU acceleration when available."""
    target_model = os.path.join(OUTPUT_DIR, f"best_model_{experiment_id}_{rnn_type}.pt")
    if os.path.exists(target_model):
        cmd = [
            sys.executable,
            os.path.join(SCRIPT_DIR, "evaluate.py"),
            "evaluate",
            "--checkpoint",
            target_model,
        ]
        env = os.environ.copy()
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError:
            print(f"⚠️ Evaluation failed for {experiment_id}.")


def sync_ledger_to_token_type(token_type):
    """Parses master ledger and segregates metrics into isolated study ledger files."""
    global_ledger = os.path.join(
        REPO_ROOT, f"evaluation_ledger_{token_type}.json"
    )
    if not os.path.exists(global_ledger):
        return

    try:
        with open(global_ledger, "r", encoding="utf-8") as f:
            g_data = json.load(f)

        remaining_g_data = {}
        for k, v in g_data.items():
            if k.upper().startswith(token_type.upper()):
                match = re.search(r"_(A|B|C|D|E)\d*|_(PIVOT)", k.upper())
                study_suffix = (
                    match.group(1) or match.group(2) if match else "MISC"
                )

                study_ledger_path = os.path.join(
                    REPO_ROOT, f"evaluation_ledger_{token_type}_{study_suffix}.json"
                )

                study_data = {}
                if os.path.exists(study_ledger_path):
                    try:
                        with open(
                            study_ledger_path, "r", encoding="utf-8"
                        ) as sf:
                            study_data = json.load(sf)
                    except Exception:
                        study_data = {}

                study_data[k] = v
                with open(study_ledger_path, "w", encoding="utf-8") as sf:
                    json.dump(study_data, sf, indent=4)
            else:
                remaining_g_data[k] = v

        if remaining_g_data:
            with open(global_ledger, "w", encoding="utf-8") as f:
                json.dump(remaining_g_data, f, indent=4)
        else:
            if os.path.exists(global_ledger):
                os.remove(global_ledger)
    except Exception as e:
        print(f"⚠️ Error occurred breaking ledger down to study files: {e}")


def get_best_hyperparameters(stage, token_type, rnn_type=None):
    """Parses validation loss metrics from hyperparameter tuning sweeps."""
    csv_path = os.path.join(
        REPO_ROOT, f"tuning_results_{token_type}_{stage}.csv"
    )
    profile = config.get("profiles", {}).get(token_type, {})
    default_args = [
        "--lr",
        str(profile.get("lr", 0.001)),
        "--dropout",
        str(profile.get("dropout", 0.3)),
        "--emb_dim",
        str(profile.get("emb_dim", 256)),
        "--hidden_dim",
        str(profile.get("hidden_dim", 512)),
    ]

    if not os.path.exists(csv_path):
        tune_configs = glob.glob(
            os.path.join(
                OUTPUT_DIR, f"best_config_TUNE_{token_type.upper()}_*.json"
            )
        )
        if tune_configs:
            best_loss = float("inf")
            best_cfg = None
            for cfg_f in tune_configs:
                try:
                    with open(cfg_f, "r", encoding="utf-8") as f:
                        c = json.load(f)
                    if (
                        rnn_type
                        and c.get("rnn_type", "").upper() != rnn_type.upper()
                    ):
                        continue
                    v_loss = float(
                        c.get("best_val_loss", c.get("val_loss", 999.0))
                    )
                    if v_loss < best_loss:
                        best_loss = v_loss
                        best_cfg = c
                except Exception:
                    pass
            if best_cfg:
                lr = best_cfg.get("lr", profile.get("lr", 0.001))
                dropout = best_cfg.get("dropout", profile.get("dropout", 0.3))
                emb_dim = int(
                    best_cfg.get("emb_dim", profile.get("emb_dim", 256))
                )
                hidden_dim = int(
                    best_cfg.get("hidden_dim", profile.get("hidden_dim", 512))
                )
                print(
                    "🎯 Optimization Checkpoint Found! Applying Tuned"
                    f" Parameters: --lr {lr} --dropout {dropout} --emb_dim"
                    f" {emb_dim} --hidden_dim {hidden_dim}"
                )
                return [
                    "--lr",
                    str(lr),
                    "--dropout",
                    str(dropout),
                    "--emb_dim",
                    str(emb_dim),
                    "--hidden_dim",
                    str(hidden_dim),
                ]

        print(
            f"ℹ️ Tuning ledger missing at {csv_path}. Falling back to standard"
            " profiles."
        )
        return default_args

    try:
        df = pd.read_csv(csv_path)
        valid = df[df["status"].astype(str).str.strip() == "Success"].copy()
        if valid.empty:
            print(
                f"⚠️ Tuning file {csv_path} contains no successful sweeps."
                " Falling back to defaults."
            )
            return default_args

        if rnn_type and "rnn_type" in valid.columns:
            cell_runs = valid[
                valid["rnn_type"].astype(str).str.upper().str.strip()
                == rnn_type.upper().strip()
            ]
            if not cell_runs.empty:
                valid = cell_runs

        valid["val_loss"] = pd.to_numeric(valid["val_loss"], errors="coerce")
        valid = valid.dropna(subset=["val_loss"])

        if valid.empty:
            print(
                "⚠️ No valid numerical optimization scores for"
                f" {rnn_type or 'all'}. Falling back to defaults."
            )
            return default_args

        valid = valid.sort_values(by="val_loss", ascending=True)
        best_run = valid.iloc[0]

        lr = best_run.get("learning_rate", profile.get("lr", 0.001))
        dropout = best_run.get("dropout", profile.get("dropout", 0.3))
        emb_dim = int(
            float(best_run.get("emb_dim", profile.get("emb_dim", 256)))
        )
        hidden_dim = int(
            float(best_run.get("hidden_dim", profile.get("hidden_dim", 512)))
        )

        print(
            "🎯 Optimization Checkpoint Found! Applying Tuned Parameters: --lr"
            f" {lr} --dropout {dropout} --emb_dim {emb_dim} --hidden_dim"
            f" {hidden_dim}"
        )
        return [
            "--lr",
            str(lr),
            "--dropout",
            str(dropout),
            "--emb_dim",
            str(emb_dim),
            "--hidden_dim",
            str(hidden_dim),
        ]

    except Exception as e:
        print(
            "⚠️ Hyperparameter parser interception exception:"
            f" {e}. Default parameter profile applied."
        )
        return default_args


def get_best_empirical_settings(token_type):
    """Retrieves top-performing empirical architectural settings from past study runs."""
    profile = config.get("profiles", {}).get(token_type, {})
    defaults = {
        "rnn_type": "LSTM",
        "bidirectional": "True",
        "embedding_source": "scratch",
        "freeze_emb": "False",
        "attention_type": "none",
        "emb_dim": str(profile.get("emb_dim", 256)),
    }

    ledger = {}
    pattern = os.path.join(REPO_ROOT, f"evaluation_ledger_{token_type}_*.json")
    for filepath in glob.glob(pattern):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                ledger.update(json.load(f))
        except Exception:
            pass

    pattern_cfg = os.path.join(
        OUTPUT_DIR, f"best_config_{token_type.upper()}_*.json"
    )
    for filepath in glob.glob(pattern_cfg):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                cdata = json.load(f)
            exp_key = cdata.get(
                "experiment",
                os.path.basename(filepath)
                .replace("best_config_", "")
                .split(".json")[0],
            )
            if exp_key not in ledger:
                ledger[exp_key] = cdata
        except Exception:
            pass

    if not ledger:
        return defaults
    try:
        prefix = token_type.upper()

        def get_composite_score(node):
            metrics = node.get("metrics", {})
            bleu = float(
                metrics.get("overall_corpus_bleu", node.get("bleu", 0.0))
            )
            meteor = float(metrics.get("mean_meteor", node.get("meteor", 0.0)))
            if bleu > 0 or meteor > 0:
                return bleu + (meteor * 100.0)
            val_loss = float(
                node.get("best_val_loss", node.get("val_loss", 999.0))
            )
            return -val_loss

        best_a = -float("inf")
        for exp in [f"{prefix}_A{i}" for i in range(1, 7)]:
            matching_keys = [
                k for k in ledger if k == exp or k.startswith(f"{exp}_")
            ]
            for k in matching_keys:
                score = get_composite_score(ledger[k])
                if score > best_a:
                    best_a = score
                    defaults["rnn_type"] = ledger[k].get(
                        "rnn_type", defaults["rnn_type"]
                    )
                    defaults["bidirectional"] = str(
                        ledger[k].get("bidirectional", defaults["bidirectional"])
                    )
                    if "emb_dim" in ledger[k]:
                        defaults["emb_dim"] = str(ledger[k]["emb_dim"])

        best_b = -float("inf")
        for exp in [f"{prefix}_B{i}" for i in range(1, 13)]:
            matching_keys = [
                k for k in ledger if k == exp or k.startswith(f"{exp}_")
            ]
            for k in matching_keys:
                score = get_composite_score(ledger[k])
                if score > best_b:
                    best_b = score
                    defaults["embedding_source"] = ledger[k].get(
                        "embedding_source", defaults["embedding_source"]
                    )
                    defaults["freeze_emb"] = str(
                        ledger[k].get("freeze_emb", defaults["freeze_emb"])
                    )
                    if "emb_dim" in ledger[k]:
                        defaults["emb_dim"] = str(ledger[k]["emb_dim"])
                    defaults["rnn_type"] = ledger[k].get(
                        "rnn_type", defaults["rnn_type"]
                    )
                    defaults["bidirectional"] = str(
                        ledger[k].get("bidirectional", defaults["bidirectional"])
                    )

        best_c = -float("inf")
        for exp in [f"{prefix}_C{i}" for i in range(1, 7)]:
            matching_keys = [
                k for k in ledger if k == exp or k.startswith(f"{exp}_")
            ]
            for k in matching_keys:
                score = get_composite_score(ledger[k])
                if score > best_c:
                    best_c = score
                    defaults["attention_type"] = ledger[k].get(
                        "attention_type", defaults["attention_type"]
                    )
                    defaults["rnn_type"] = ledger[k].get(
                        "rnn_type", defaults["rnn_type"]
                    )
                    defaults["bidirectional"] = str(
                        ledger[k].get("bidirectional", defaults["bidirectional"])
                    )
                    if "emb_dim" in ledger[k]:
                        defaults["emb_dim"] = str(ledger[k]["emb_dim"])
    except Exception:
        pass
    return defaults


def execute_preprocessing(token_type="word", mock_mode=False):
    """Executes offline dataset preprocessing and binary caching routines."""
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "preprocess.py"),
        "--token_type",
        token_type,
    ]
    if mock_mode:
        cmd.append("--mock")

    print(
        "⚡ Running preprocessing routine"
        f" (token_type={token_type}, mock={mock_mode})..."
    )
    subprocess.run(cmd, check=True)


def execute_tuning(
    stage="coarse", token_type="word", epochs=4, num_trials=15, configs_per_rnn=None
):
    """Executes hyperparameter tuning sweeps evenly across RNN, GRU, and LSTM cell types."""
    print(
        "\n"
        + "═" * 75
        + f"\n🔍 RUNNING HYPERPARAMETER TUNING ({stage.upper()} -"
        f" {token_type.upper()} | {epochs} Epochs)\n"
        + "═" * 75
    )

    lrs = [0.0003, 0.0005, 0.001, 0.002]
    dropouts = [0.2, 0.3, 0.4]
    emb_dims = [128, 256, 512] if token_type == "word" else [32, 64, 128]
    hidden_dims = [256, 512, 1024]
    rnn_types = ["LSTM", "GRU", "RNN"]

    batch_size = get_batch_size("TUNE", token_type)
    results_csv = os.path.join(
        REPO_ROOT, f"tuning_results_{token_type}_{stage}.csv"
    )

    fieldnames = [
        "run_id",
        "stage",
        "token_type",
        "rnn_type",
        "learning_rate",
        "dropout",
        "emb_dim",
        "hidden_dim",
        "val_loss",
        "status",
    ]

    if not os.path.exists(results_csv):
        with open(results_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    random.seed(42)

    # Determine balanced trials per RNN architecture (e.g. 3 or 4 per cell type)
    if configs_per_rnn is not None:
        trials_per_rnn = configs_per_rnn
    else:
        trials_per_rnn = max(1, num_trials // len(rnn_types))

    selected_trials = []
    for rnn in rnn_types:
        combos = list(itertools.product(lrs, dropouts, emb_dims, hidden_dims, [rnn]))
        random.shuffle(combos)
        selected_trials.extend(combos[:trials_per_rnn])

    # Fill any remaining quota round-robin if num_trials exceeds total selected
    if len(selected_trials) < num_trials and configs_per_rnn is None:
        all_combos = list(itertools.product(lrs, dropouts, emb_dims, hidden_dims, rnn_types))
        remaining = [c for c in all_combos if c not in selected_trials]
        random.shuffle(remaining)
        selected_trials.extend(remaining[:num_trials - len(selected_trials)])

    print(
        f"📋 Selected {len(selected_trials)} trials total "
        f"({trials_per_rnn} configs tested per cell type across {rnn_types})."
    )

    best_loss = float("inf")
    best_params = None

    for idx, (lr, drop, emb_d, hid_d, rnn) in enumerate(selected_trials, 1):
        exp_id = f"TUNE_{token_type.upper()}_{stage.upper()}_{idx}"
        print(
            f"\n🧪 [Trial {idx}/{len(selected_trials)}] -> LR={lr}, Dropout={drop},"
            f" Emb={emb_d}, Hidden={hid_d}, Cell={rnn}"
        )

        cmd = [
            "--experiment",
            exp_id,
            "--rnn_type",
            rnn,
            "--token_type",
            token_type,
            "--lr",
            str(lr),
            "--dropout",
            str(drop),
            "--emb_dim",
            str(emb_d),
            "--hidden_dim",
            str(hid_d),
            "--batch_size",
            batch_size,
            "--epochs",
            str(epochs),
            "--src_lang",
            "en",
            "--trg_lang",
            "de",
        ]

        try:
            run_cmd(cmd)
            status = "Success"

            json_cfg_file = os.path.join(
                OUTPUT_DIR, f"best_config_{exp_id}_{rnn}.json"
            )
            val_loss = float("inf")
            if os.path.exists(json_cfg_file):
                with open(json_cfg_file, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    val_loss = float(
                        cdata.get("best_val_loss", cdata.get("val_loss", 999.0))
                    )

            if val_loss < best_loss:
                best_loss = val_loss
                best_params = {
                    "lr": lr,
                    "dropout": drop,
                    "emb_dim": emb_d,
                    "hidden_dim": hid_d,
                    "rnn_type": rnn,
                    "val_loss": val_loss,
                }

        except Exception as e:
            print(f"⚠️ Trial {idx} failed: {e}")
            status = "Failed"
            val_loss = float("inf")

        with open(results_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({
                "run_id": exp_id,
                "stage": stage,
                "token_type": token_type,
                "rnn_type": rnn,
                "learning_rate": lr,
                "dropout": drop,
                "emb_dim": emb_d,
                "hidden_dim": hid_d,
                "val_loss": val_loss,
                "status": status,
            })

    if best_params:
        summary_json = os.path.join(
            OUTPUT_DIR,
            f"best_config_TUNE_{token_type.upper()}_{stage.upper()}.json",
        )
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=4)
        print(
            f"\n🏆 Best Tuning Parameters Saved -> {summary_json} (Val"
            f" Loss: {best_loss:.4f})"
        )


def run_automated_post_processing(token_type, rnn_type):
    """Executes evaluation aggregation, pivot evaluation, and attention heatmap generation."""
    env = os.environ.copy()

    try:
        subprocess.run(
            [
                sys.executable,
                os.path.join(SCRIPT_DIR, "evaluate.py"),
                "evaluate",
                "--token_type",
                token_type,
            ],
            check=True,
            env=env,
        )
    except Exception:
        pass

    de_en_model = os.path.join(
        OUTPUT_DIR, f"best_model_{token_type.upper()}_D2_{rnn_type}.pt"
    )
    en_sv_model = os.path.join(
        OUTPUT_DIR, f"best_model_{token_type.upper()}_E1_{rnn_type}.pt"
    )
    if os.path.exists(de_en_model) and os.path.exists(en_sv_model):
        try:
            subprocess.run(
                [
                    sys.executable,
                    os.path.join(SCRIPT_DIR, "pivot.py"),
                    "--de_en_model",
                    de_en_model,
                    "--en_sv_model",
                    en_sv_model,
                    "--text",
                    "maschinelles lernen macht unglaublichen spass",
                ],
                check=True,
                env=env,
            )
        except Exception:
            pass

        print(
            "\n📊 Launching Formal Quantitative Pivot Dataset Evaluation (DE ➔"
            " EN ➔ SV)..."
        )
        try:
            subprocess.run(
                [
                    sys.executable,
                    os.path.join(SCRIPT_DIR, "pivot.py"),
                    "--de_en_model",
                    de_en_model,
                    "--en_sv_model",
                    en_sv_model,
                    "--evaluate",
                    "--token_type",
                    token_type,
                    "--experiment",
                    f"{token_type.upper()}_PIVOT",
                ],
                check=True,
                env=env,
            )
            sync_ledger_to_token_type(token_type)
        except Exception as e:
            print(
                "⚠️ Quantitative pivot dataset evaluation interrupted or"
                f" unsupported: {e}"
            )

    try:
        generate_all_reports(token_type)
    except Exception as e:
        print(f"⚠️ Error compiling reports: {e}")

    attn_model = os.path.join(
        OUTPUT_DIR, f"best_model_{token_type.upper()}_C4_{rnn_type}.pt"
    )
    if not os.path.exists(attn_model):
        attn_model = os.path.join(
            OUTPUT_DIR, f"best_model_{token_type.upper()}_C3_{rnn_type}.pt"
        )

    if os.path.exists(attn_model):
        try:
            visualize_attention(attn_model)
        except Exception as e:
            print(f"⚠️ Error rendering attention heatmap: {e}")


def execute_study_a(epochs, token_type, eval_queue: AsyncEvaluationQueue):
    """Executes Study A: Recurrent Architecture Benchmarking (RNN vs GRU vs LSTM, Uni vs Bi)."""
    configs = [
        ("A1", "RNN", "False"),
        ("A2", "RNN", "True"),
        ("A3", "GRU", "False"),
        ("A4", "GRU", "True"),
        ("A5", "LSTM", "False"),
        ("A6", "LSTM", "True"),
    ]
    batch_size = get_batch_size("A", token_type)
    for exp, cell, bidi in configs:
        exp_id = f"{token_type.upper()}_{exp}"
        hparams = get_best_hyperparameters("coarse", token_type, rnn_type=cell)

        emb_dim = (
            hparams[hparams.index("--emb_dim") + 1]
            if "--emb_dim" in hparams
            else "256"
        )
        hidden_dim = (
            hparams[hparams.index("--hidden_dim") + 1]
            if "--hidden_dim" in hparams
            else "512"
        )

        print_study_model_and_batch_info(
            study_name="Study A (Architecture Benchmarking)",
            exp_id=exp_id,
            token_type=token_type,
            rnn_type=cell,
            bidirectional=bidi,
            attention_type="none",
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
        )

        ckpt_path = os.path.join(OUTPUT_DIR, f"best_model_{exp_id}_{cell}.pt")
        cfg_path = os.path.join(OUTPUT_DIR, f"best_config_{exp_id}_{cell}.json")

        if not is_cache_valid(ckpt_path, cfg_path):
            run_cmd(
                hparams
                + [
                    "--experiment",
                    exp_id,
                    "--rnn_type",
                    cell,
                    "--bidirectional",
                    bidi,
                    "--token_type",
                    token_type,
                    "--batch_size",
                    batch_size,
                    "--epochs",
                    str(epochs),
                    "--src_lang",
                    "en",
                    "--trg_lang",
                    "de",
                ]
            )

        eval_queue.submit_evaluation(exp_id, cell, token_type)

    eval_queue.sync_study()


def execute_study_b(
    epochs, rnn_type, bidirectional, token_type, eval_queue: AsyncEvaluationQueue
):
    """Executes Study B: Input Embedding Representation & Dimensionality Benchmarking (EN -> DE)."""
    hparams = get_best_hyperparameters("coarse", token_type, rnn_type=rnn_type)
    configs = (
        [
            ("B1", "scratch", "False", "256"),
            ("B2", "word2vec", "True", "300"),
            ("B3", "word2vec", "False", "300"),
            ("B4", "scratch", "True", "256"),
            ("B5", "glove", "True", "300"),
            ("B6", "glove", "False", "300"),
        ]
        if token_type == "word"
        else [
            ("B7", "scratch", "False", "32"),
            ("B8", "scratch", "False", "64"),
            ("B9", "scratch", "False", "128"),
            ("B10", "onehot", "True", "128"),
        ]
    )
    batch_size = get_batch_size("B", token_type)
    hidden_dim = (
        hparams[hparams.index("--hidden_dim") + 1]
        if "--hidden_dim" in hparams
        else "512"
    )

    for exp, src, freeze, emb_dim in configs:
        exp_id = f"{token_type.upper()}_{exp}"

        print_study_model_and_batch_info(
            study_name="Study B (Embedding Representation Analysis - EN->DE)",
            exp_id=exp_id,
            token_type=token_type,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            attention_type="none",
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
        )

        ckpt_path = os.path.join(OUTPUT_DIR, f"best_model_{exp_id}_{rnn_type}.pt")
        cfg_path = os.path.join(
            OUTPUT_DIR, f"best_config_{exp_id}_{rnn_type}.json"
        )

        if not is_cache_valid(ckpt_path, cfg_path):
            run_cmd(
                hparams
                + [
                    "--experiment",
                    exp_id,
                    "--rnn_type",
                    rnn_type,
                    "--bidirectional",
                    bidirectional,
                    "--token_type",
                    token_type,
                    "--embedding_source",
                    "scratch" if src == "onehot" else src,
                    "--freeze_emb",
                    freeze,
                    "--emb_dim",
                    emb_dim,
                    "--batch_size",
                    batch_size,
                    "--epochs",
                    str(epochs),
                    "--src_lang",
                    "en",
                    "--trg_lang",
                    "de",
                ]
            )

        eval_queue.submit_evaluation(exp_id, rnn_type, token_type)

    eval_queue.sync_study()


def execute_study_c(
    epochs,
    token_type,
    rnn_type,
    bidirectional,
    embedding_source,
    freeze_emb,
    emb_dim,
    eval_queue: AsyncEvaluationQueue,
):
    """Executes Study C: Attention Mechanism Optimization (Luong vs Bahdanau vs None) (EN -> DE)."""
    hparams = get_best_hyperparameters("coarse", token_type, rnn_type=rnn_type)
    configs = [
        ("C1", rnn_type, "none", "False"),
        ("C2", rnn_type, "none", bidirectional),
        ("C3", rnn_type, "luong", bidirectional),
        ("C4", rnn_type, "bahdanau", bidirectional),
        ("C5", "RNN", "luong", "True"),
        ("C6", "RNN", "bahdanau", "True"),
    ]

    batch_size = get_batch_size("C", token_type)
    hidden_dim = (
        hparams[hparams.index("--hidden_dim") + 1]
        if "--hidden_dim" in hparams
        else "512"
    )

    for exp, cell, attn, bidi in configs:
        exp_id = f"{token_type.upper()}_{exp}"

        print_study_model_and_batch_info(
            study_name="Study C (Attention Mechanism Optimization - EN->DE)",
            exp_id=exp_id,
            token_type=token_type,
            rnn_type=cell,
            bidirectional=bidi,
            attention_type=attn,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
        )

        ckpt_path = os.path.join(OUTPUT_DIR, f"best_model_{exp_id}_{cell}.pt")
        cfg_path = os.path.join(OUTPUT_DIR, f"best_config_{exp_id}_{cell}.json")

        if not is_cache_valid(ckpt_path, cfg_path):
            cmd_hparams = (
                get_best_hyperparameters("coarse", token_type, rnn_type=cell)
                if cell == "RNN"
                else hparams
            )
            run_cmd(
                cmd_hparams
                + [
                    "--experiment",
                    exp_id,
                    "--rnn_type",
                    cell,
                    "--attention_type",
                    attn,
                    "--bidirectional",
                    bidi,
                    "--token_type",
                    token_type,
                    "--embedding_source",
                    embedding_source,
                    "--freeze_emb",
                    freeze_emb,
                    "--emb_dim",
                    emb_dim,
                    "--batch_size",
                    batch_size,
                    "--epochs",
                    str(epochs),
                    "--src_lang",
                    "en",
                    "--trg_lang",
                    "de",
                ]
            )

        eval_queue.submit_evaluation(exp_id, cell, token_type)

    eval_queue.sync_study()


def execute_study_d(
    epochs,
    token_type,
    rnn_type,
    bidirectional,
    embedding_source,
    freeze_emb,
    attention_type,
    emb_dim,
    eval_queue: AsyncEvaluationQueue,
):
    """Executes Study D: Translation Direction Optimization (EN ➔ DE vs DE ➔ EN)."""
    hparams = get_best_hyperparameters("fine", token_type, rnn_type=rnn_type)
    configs = [("D1", "en", "de"), ("D2", "de", "en")]
    batch_size = get_batch_size("D", token_type)
    hidden_dim = (
        hparams[hparams.index("--hidden_dim") + 1]
        if "--hidden_dim" in hparams
        else "512"
    )

    for exp, src, trg in configs:
        exp_id = f"{token_type.upper()}_{exp}"

        print_study_model_and_batch_info(
            study_name="Study D (Language Direction Optimization)",
            exp_id=exp_id,
            token_type=token_type,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            attention_type=attention_type,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
        )

        ckpt_path = os.path.join(
            OUTPUT_DIR, f"best_model_{exp_id}_{rnn_type}.pt"
        )
        cfg_path = os.path.join(
            OUTPUT_DIR, f"best_config_{exp_id}_{rnn_type}.json"
        )

        if not is_cache_valid(ckpt_path, cfg_path):
            run_cmd(
                hparams
                + [
                    "--experiment",
                    exp_id,
                    "--rnn_type",
                    rnn_type,
                    "--attention_type",
                    attention_type,
                    "--bidirectional",
                    bidirectional,
                    "--token_type",
                    token_type,
                    "--embedding_source",
                    embedding_source,
                    "--freeze_emb",
                    freeze_emb,
                    "--emb_dim",
                    emb_dim,
                    "--src_lang",
                    src,
                    "--trg_lang",
                    trg,
                    "--batch_size",
                    batch_size,
                    "--epochs",
                    str(epochs),
                ]
            )

        eval_queue.submit_evaluation(exp_id, rnn_type, token_type)

    eval_queue.sync_study()


def execute_study_e(
    epochs,
    token_type,
    rnn_type,
    bidirectional,
    embedding_source,
    freeze_emb,
    attention_type,
    emb_dim,
    eval_queue: AsyncEvaluationQueue,
):
    """Executes Study E: Generalization & Swedish Pivot Channel Construction (EN ➔ SV)."""
    hparams = get_best_hyperparameters("fine", token_type, rnn_type=rnn_type)
    configs = [("E1", "en", "sv")]
    batch_size = get_batch_size("E", token_type)
    hidden_dim = (
        hparams[hparams.index("--hidden_dim") + 1]
        if "--hidden_dim" in hparams
        else "512"
    )

    for exp, src, trg in configs:
        exp_id = f"{token_type.upper()}_{exp}"

        print_study_model_and_batch_info(
            study_name="Study E (Generalization & Pivot Pipeline)",
            exp_id=exp_id,
            token_type=token_type,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            attention_type=attention_type,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
        )

        ckpt_path = os.path.join(
            OUTPUT_DIR, f"best_model_{exp_id}_{rnn_type}.pt"
        )
        cfg_path = os.path.join(
            OUTPUT_DIR, f"best_config_{exp_id}_{rnn_type}.json"
        )

        if not is_cache_valid(ckpt_path, cfg_path):
            run_cmd(
                hparams
                + [
                    "--experiment",
                    exp_id,
                    "--rnn_type",
                    rnn_type,
                    "--attention_type",
                    attention_type,
                    "--bidirectional",
                    bidirectional,
                    "--token_type",
                    token_type,
                    "--embedding_source",
                    embedding_source,
                    "--freeze_emb",
                    freeze_emb,
                    "--emb_dim",
                    emb_dim,
                    "--src_lang",
                    src,
                    "--trg_lang",
                    trg,
                    "--batch_size",
                    batch_size,
                    "--epochs",
                    str(epochs),
                ]
            )

        eval_queue.submit_evaluation(exp_id, rnn_type, token_type)

    eval_queue.sync_study()


def main():
    setup_logging(log_filename="run_studies.log", log_dir=OUTPUT_DIR)

    parser = argparse.ArgumentParser(
        description="Master Empirical NMT Orchestrator Interface"
    )
    parser.add_argument(
        "--study",
        type=str,
        default="all",
        choices=["all", "A", "B", "C", "D", "E", "tune", "fine_tune", "postprocess"],
        help="Specify study suite to run or execute 'all'",
    )
    parser.add_argument(
        "--token_type",
        type=str,
        default="word",
        choices=["word", "char"],
        help="Tokenization level",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Execute in rapid mock mode with small synthetic sample dataset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epoch scheduling across all stages with explicit integer",
    )
    parser.add_argument(
        "--tune_stage",
        type=str,
        default="coarse",
        choices=["coarse", "fine"],
        help="Hyperparameter tuning stage",
    )
    parser.add_argument(
        "--tune_trials",
        type=int,
        default=12,
        help="Total number of hyperparameter search trials (e.g. 12 = 4 trials per cell type)",
    )
    parser.add_argument(
        "--configs_per_rnn",
        type=int,
        default=None,
        help="Explicitly force N trials per cell type (e.g. 3 or 4 per cell type)",
    )
    parser.add_argument(
        "--no_preprocess",
        action="store_true",
        help="Skip data preprocessing step if cached dataset files exist",
    )

    args = parser.parse_args()

    # Dynamic epoch schedule definitions
    TUNE_1_EPOCHS = args.epochs if args.epochs is not None else 4
    TUNE_2_EPOCHS = args.epochs if args.epochs is not None else 6
    STUDY_ABC_EPOCHS = args.epochs if args.epochs is not None else 6
    STUDY_DE_EPOCHS = args.epochs if args.epochs is not None else 6

    print("\n" + "═" * 80)
    print("🚀 NMT PERFORMANCE INFRASTRUCTURE ORCHESTRATOR INITIALIZED")
    print(
        f"   Mode: {args.study.upper()} | Token Level: {args.token_type.upper()}"
        f" | Dynamic Epoch Strategy: [Tune1: {TUNE_1_EPOCHS}, Studies A-C: {STUDY_ABC_EPOCHS}, Tune2: {TUNE_2_EPOCHS}, Studies D-E: {STUDY_DE_EPOCHS}]"
        f" | GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}"
    )
    print("═" * 80 + "\n")

    if not args.no_preprocess and args.study != "postprocess":
        execute_preprocessing(token_type=args.token_type, mock_mode=args.mock)

    eval_queue = AsyncEvaluationQueue(max_workers=2)

    # 1. First Tuning Pass (Coarse Sweep) - 4 Epochs
    if args.study in ["tune", "all"] and args.tune_stage == "coarse":
        execute_tuning(
            stage="coarse",
            token_type=args.token_type,
            epochs=TUNE_1_EPOCHS,
            num_trials=args.tune_trials,
            configs_per_rnn=args.configs_per_rnn,
        )

    # 2. Studies A, B, C - 6 Epochs
    if args.study in ["all", "A"]:
        execute_study_a(STUDY_ABC_EPOCHS, args.token_type, eval_queue)

    best_settings = get_best_empirical_settings(args.token_type)

    if args.study in ["all", "B"]:
        execute_study_b(
            STUDY_ABC_EPOCHS,
            best_settings["rnn_type"],
            best_settings["bidirectional"],
            args.token_type,
            eval_queue,
        )

    best_settings = get_best_empirical_settings(args.token_type)

    if args.study in ["all", "C"]:
        execute_study_c(
            STUDY_ABC_EPOCHS,
            args.token_type,
            best_settings["rnn_type"],
            best_settings["bidirectional"],
            best_settings["embedding_source"],
            best_settings["freeze_emb"],
            best_settings["emb_dim"],
            eval_queue,
        )

    # 3. Second Tuning Pass (Fine Sweep) - 6 Epochs
    if args.study in ["all", "fine_tune"] or (args.study == "tune" and args.tune_stage == "fine"):
        execute_tuning(
            stage="fine",
            token_type=args.token_type,
            epochs=TUNE_2_EPOCHS,
            num_trials=args.tune_trials,
            configs_per_rnn=args.configs_per_rnn,
        )

    best_settings = get_best_empirical_settings(args.token_type)

    # 4. Studies D and E - 6 Epochs
    if args.study in ["all", "D"]:
        execute_study_d(
            STUDY_DE_EPOCHS,
            args.token_type,
            best_settings["rnn_type"],
            best_settings["bidirectional"],
            best_settings["embedding_source"],
            best_settings["freeze_emb"],
            best_settings["attention_type"],
            best_settings["emb_dim"],
            eval_queue,
        )

    if args.study in ["all", "E"]:
        execute_study_e(
            STUDY_DE_EPOCHS,
            args.token_type,
            best_settings["rnn_type"],
            best_settings["bidirectional"],
            best_settings["embedding_source"],
            best_settings["freeze_emb"],
            best_settings["attention_type"],
            best_settings["emb_dim"],
            eval_queue,
        )

    eval_queue.shutdown()

    if args.study in ["all", "postprocess"]:
        run_automated_post_processing(args.token_type, best_settings["rnn_type"])

    print("\n" + "═" * 80)
    print(
        "🎉 MASTER ORCHESTRATION PIPELINE COMPLETED SUCCESSFULLY ON GPU/CPU"
        " CLUSTER"
    )
    print("═" * 80 + "\n")


if __name__ == "__main__":
    main()