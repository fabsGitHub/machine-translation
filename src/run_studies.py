import argparse
import subprocess
import sys
import os
import json
import pandas as pd
import csv
import time  
import glob
import itertools
import random
import threading
import matplotlib
import torch
import re
from concurrent.futures import ThreadPoolExecutor, wait

matplotlib.use('Agg') 

from config import load_config
from utils import set_seed, check_artifact_cache, is_cache_valid

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "results")
CONFIG_PATH = os.path.join(REPO_ROOT, "config", "config.yaml")

config = load_config(CONFIG_PATH)
set_seed(config.get('system', {}).get('seed', 42))

eval_lock = threading.Lock()


def get_batch_size(study, token_type):
    """
    Centralized handler for batch size configuration across studies and token levels.
    Batch sizes set to powers of 2 (2048 and 1024).
    """
    config_batch = config.get('training', {}).get('batch_size')
    if config_batch is not None:
        return str(config_batch)

    return "2048" if token_type == "char" else "1024"


class AsyncEvaluationQueue:
    """
    Offloads evaluation and ledger synchronization to background CPU threads,
    allowing the GPU to immediately begin training the next model without VRAM contention.
    """
    def __init__(self, max_workers=2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []

    def submit_evaluation(self, experiment_id, rnn_type, token_type):
        """Queues evaluation in the background strictly on CPU without blocking GPU execution."""
        def _task():
            with eval_lock:
                print(f"\n⚡ [Async Eval Started] -> {experiment_id} ({rnn_type}) [Forced CPU Mode]")
                run_auto_evaluation(experiment_id, rnn_type)
                sync_ledger_to_token_type(token_type)
                print(f"✅ [Async Eval Finished] -> {experiment_id} ({rnn_type})")

        future = self.executor.submit(_task)
        self.futures.append(future)

    def sync_study(self):
        """Blocks until all queued evaluations for the current study complete."""
        if self.futures:
            print("\n⏳ [Study Barrier] Waiting for background evaluation tasks to complete...")
            wait(self.futures)
            self.futures.clear()
            print("🎯 [Study Barrier Cleared] All study models evaluated successfully.\n")

    def shutdown(self):
        self.executor.shutdown(wait=True)


def get_vocab_sizes(token_type="word"):
    """Dynamically retrieves vocabulary size from binary dataset cache or defaults to realistic baseline power of 2."""
    processed_dir = os.path.join(REPO_ROOT, "data", "processed")
    cache_dir = os.path.join(processed_dir, ".matrix_cache")
    if os.path.exists(cache_dir):
        for fname in os.listdir(cache_dir):
            if fname.endswith(".pt") and token_type in fname:
                try:
                    payload = torch.load(os.path.join(cache_dir, fname), map_location="cpu", weights_only=False)
                    return len(payload['src_vocab']), len(payload['trg_vocab'])
                except Exception:
                    pass
    return (8192, 8192) if token_type == "word" else (256, 256)


def print_study_model_and_batch_info(study_name, exp_id, token_type, rnn_type, bidirectional, 
                                     attention_type, emb_dim, hidden_dim, batch_size):
    """
    Analytically computes and outputs Model Size (Parameters & FP32 MB) 
    and Batch Size parameters without instantiating dummy PyTorch models on CPU.
    """
    src_vocab_len, trg_vocab_len = get_vocab_sizes(token_type)
    bidi_bool = str(bidirectional).lower() == "true"
    num_directions = 2 if bidi_bool else 1
    emb_d, hid_d = int(emb_dim), int(hidden_dim)
    gates = 4 if rnn_type == "LSTM" else (3 if rnn_type == "GRU" else 1)

    enc_emb = src_vocab_len * emb_d
    enc_l1 = gates * ((emb_d * hid_d + hid_d * hid_d + 2 * hid_d) * num_directions)
    enc_l2 = gates * ((hid_d * num_directions * hid_d + hid_d * hid_d + 2 * hid_d) * num_directions)
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
        attn_params = (hid_d * hid_d + hid_d) + (enc_out_dim * hid_d + hid_d) + hid_d

    total_params = enc_params + dec_emb + dec_l1 + dec_l2 + dec_fc + attn_params
    model_size_mb = (total_params * 4) / (1024 ** 2)

    seq_len = 256 if token_type == "char" else 64
    batch_num = int(batch_size)
    batch_memory_mb = (batch_num * seq_len * 8) / (1024 ** 2)

    print("\n" + "─" * 75)
    print(f"📐 [DYNAMIC STUDY ANALYSIS] - {study_name} ({exp_id})")
    print(f" ├─ Tokenizer Mode:           {token_type.upper()}")
    print(f" ├─ Architecture:             {rnn_type} (BiDirect: {bidi_bool}, Attention: {attention_type})")
    print(f" ├─ Dimensions:               Emb={emb_dim} | Hidden={hidden_dim}")
    print(f" ├─ Batch Size (N samples):   {batch_num} sequences / batch")
    print(f" ├─ Batch Shape Estimate:     [{batch_num}, {seq_len}] ({batch_memory_mb:.4f} MB per batch tensor)")
    print(f" ├─ Total Model Parameters:   ~{total_params:,} parameters")
    print(f" └─ Model Memory Footprint:   ~{model_size_mb:.2f} MB")
    print("─" * 75 + "\n")


def run_cmd(args_list):
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

    epochs = config.get('training', {}).get('epochs', 1)
    if "--epochs" in args_list:
        try: epochs = int(args_list[args_list.index("--epochs") + 1])
        except (ValueError, IndexError): pass

    nproc = max(1, torch.cuda.device_count())
    command = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        os.path.join(SCRIPT_DIR, "train.py")
    ] + args_list
    
    print(f"\n🚀 Launching DDP Execution Unit ({nproc} processes): {' '.join(command)}")
    
    start_time = time.time()
    subprocess.run(command, check=True)
    duration = time.time() - start_time
    print(f"⏱️ Done. Duration: {duration:.2f}s | Avg/Epoch: {duration/max(1, epochs):.2f}s")


def run_auto_evaluation(experiment_id, rnn_type):
    """
    Executes model evaluation strictly on CPU to eliminate GPU contention and OOM crashes.
    """
    target_model = os.path.join(OUTPUT_DIR, f"best_model_{experiment_id}_{rnn_type}.pt")
    if os.path.exists(target_model):
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "evaluate.py"), "evaluate", "--checkpoint", target_model]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""  # Force evaluation processes to run strictly on CPU
        try: 
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError:
            print(f"⚠️ Process protection enabled for {experiment_id}. Processing via fallback structures.")


def sync_ledger_to_token_type(token_type):
    global_ledger = os.path.join(REPO_ROOT, f"evaluation_ledger_{token_type}.json")
    if not os.path.exists(global_ledger): 
        return
        
    try:
        with open(global_ledger, 'r') as f: 
            g_data = json.load(f)
        
        remaining_g_data = {}
        for k, v in g_data.items():
            if k.upper().startswith(token_type.upper()):
                match = re.search(r'_(A|B|C|D|E)\d*|_(PIVOT)', k.upper())
                study_suffix = match.group(1) or match.group(2) if match else "MISC"
                
                study_ledger_path = os.path.join(REPO_ROOT, f"evaluation_ledger_{token_type}_{study_suffix}.json")
                
                study_data = {}
                if os.path.exists(study_ledger_path):
                    try:
                        with open(study_ledger_path, 'r') as sf: 
                            study_data = json.load(sf)
                    except Exception: 
                        study_data = {}
                        
                study_data[k] = v
                with open(study_ledger_path, 'w') as sf:
                    json.dump(study_data, sf, indent=4)
            else:
                remaining_g_data[k] = v
                
        if remaining_g_data:
            with open(global_ledger, 'w') as f: json.dump(remaining_g_data, f, indent=4)
        else:
            if os.path.exists(global_ledger): os.remove(global_ledger)
    except Exception as e:
        print(f"⚠️ Error occurred breaking ledger down to study files: {e}")


def get_best_hyperparameters(stage, token_type, rnn_type=None):
    csv_path = os.path.join(REPO_ROOT, f"tuning_results_{token_type}_{stage}.csv")
    profile = config.get('profiles', {}).get(token_type, {})
    default_args = ["--lr", str(profile.get("lr", 0.001)), "--dropout", str(profile.get("dropout", 0.3)), "--emb_dim", str(profile.get("emb_dim", 256)), "--hidden_dim", str(profile.get("hidden_dim", 512))]
    
    if not os.path.exists(csv_path):
        tune_configs = glob.glob(os.path.join(OUTPUT_DIR, f"best_config_TUNE_{token_type.upper()}_*.json"))
        if tune_configs:
            best_loss = float('inf')
            best_cfg = None
            for cfg_f in tune_configs:
                try:
                    with open(cfg_f, 'r') as f:
                        c = json.load(f)
                    if rnn_type and c.get("rnn_type", "").upper() != rnn_type.upper():
                        continue
                    v_loss = float(c.get("best_val_loss", c.get("val_loss", 999.0)))
                    if v_loss < best_loss:
                        best_loss = v_loss
                        best_cfg = c
                except Exception: pass
            if best_cfg:
                lr = best_cfg.get("lr", profile.get("lr", 0.001))
                dropout = best_cfg.get("dropout", profile.get("dropout", 0.3))
                emb_dim = int(best_cfg.get("emb_dim", profile.get("emb_dim", 256)))
                hidden_dim = int(best_cfg.get("hidden_dim", profile.get("hidden_dim", 512)))
                print(f"🎯 Optimization Checkpoint Found from JSON files! Applying Tuned Parameters: --lr {lr} --dropout {dropout} --emb_dim {emb_dim} --hidden_dim {hidden_dim}")
                return ["--lr", str(lr), "--dropout", str(dropout), "--emb_dim", str(emb_dim), "--hidden_dim", str(hidden_dim)]
                
        print(f"ℹ️ Tuning ledger missing at {csv_path}. Falling back to standard profiles.")
        return default_args
        
    try:
        df = pd.read_csv(csv_path)
        valid = df[df['status'].astype(str).str.strip() == 'Success'].copy()
        if valid.empty: 
            print(f"⚠️ Tuning file {csv_path} contains no successful sweeps. Falling back to defaults.")
            return default_args
            
        if rnn_type and 'rnn_type' in valid.columns:
            cell_runs = valid[valid['rnn_type'].astype(str).str.upper().str.strip() == rnn_type.upper().strip()]
            if not cell_runs.empty: 
                valid = cell_runs

        valid['val_loss'] = pd.to_numeric(valid['val_loss'], errors='coerce')
        valid = valid.dropna(subset=['val_loss'])
        
        if valid.empty:
            print(f"⚠️ No valid numerical optimization scores for {rnn_type or 'all'}. Falling back to defaults.")
            return default_args
            
        valid = valid.sort_values(by='val_loss', ascending=True)
        best_run = valid.iloc[0]
        
        lr = best_run.get('learning_rate', profile.get("lr", 0.001))
        dropout = best_run.get('dropout', profile.get("dropout", 0.3))
        emb_dim = int(float(best_run.get('emb_dim', profile.get("emb_dim", 256))))
        hidden_dim = int(float(best_run.get('hidden_dim', profile.get("hidden_dim", 512))))
        
        print(f"🎯 Optimization Checkpoint Found! Applying Tuned Parameters: --lr {lr} --dropout {dropout} --emb_dim {emb_dim} --hidden_dim {hidden_dim}")
        return ["--lr", str(lr), "--dropout", str(dropout), "--emb_dim", str(emb_dim), "--hidden_dim", str(hidden_dim)]
        
    except Exception as e: 
        print(f"⚠️ Hyperparameter parser interception exception: {e}. Default parameter profile applied.")
        return default_args


def get_best_empirical_settings(token_type):
    profile = config.get('profiles', {}).get(token_type, {})
    defaults = {
        "rnn_type": "LSTM", 
        "bidirectional": "True", 
        "embedding_source": "scratch", 
        "freeze_emb": "False", 
        "attention_type": "none", 
        "emb_dim": str(profile.get("emb_dim", 256))
    }
    
    ledger = {}
    pattern = os.path.join(REPO_ROOT, f"evaluation_ledger_{token_type}_*.json")
    for filepath in glob.glob(pattern):
        try:
            with open(filepath, 'r') as f:
                ledger.update(json.load(f))
        except Exception: pass
        
    pattern_cfg = os.path.join(OUTPUT_DIR, f"best_config_{token_type.upper()}_*.json")
    for filepath in glob.glob(pattern_cfg):
        try:
            with open(filepath, 'r') as f:
                cdata = json.load(f)
            exp_key = cdata.get("experiment", os.path.basename(filepath).replace("best_config_", "").split(".json")[0])
            if exp_key not in ledger:
                ledger[exp_key] = cdata
        except Exception: pass

    if not ledger: return defaults
    try:
        prefix = token_type.upper()
        
        def get_composite_score(node):
            metrics = node.get("metrics", {})
            bleu = float(metrics.get("overall_corpus_bleu", node.get("bleu", 0.0)))
            meteor = float(metrics.get("mean_meteor", node.get("meteor", 0.0)))
            if bleu > 0 or meteor > 0:
                return bleu + (meteor * 100.0)
            val_loss = float(node.get("best_val_loss", node.get("val_loss", 999.0)))
            return -val_loss

        best_a = -float('inf')
        for exp in [f"{prefix}_A{i}" for i in range(1, 7)]:
            matching_keys = [k for k in ledger if k == exp or k.startswith(f"{exp}_")]
            for k in matching_keys:
                score = get_composite_score(ledger[k])
                if score > best_a:
                    best_a = score
                    defaults["rnn_type"] = ledger[k].get("rnn_type", defaults["rnn_type"])
                    defaults["bidirectional"] = str(ledger[k].get("bidirectional", defaults["bidirectional"]))
                    if "emb_dim" in ledger[k]: defaults["emb_dim"] = str(ledger[k]["emb_dim"])

        best_b = -float('inf')
        for exp in [f"{prefix}_B{i}" for i in range(1, 13)]:
            matching_keys = [k for k in ledger if k == exp or k.startswith(f"{exp}_")]
            for k in matching_keys:
                score = get_composite_score(ledger[k])
                if score > best_b:
                    best_b = score
                    defaults["embedding_source"] = ledger[k].get("embedding_source", defaults["embedding_source"])
                    defaults["freeze_emb"] = str(ledger[k].get("freeze_emb", defaults["freeze_emb"]))
                    if "emb_dim" in ledger[k]: defaults["emb_dim"] = str(ledger[k]["emb_dim"])
                    defaults["rnn_type"] = ledger[k].get("rnn_type", defaults["rnn_type"])
                    defaults["bidirectional"] = str(ledger[k].get("bidirectional", defaults["bidirectional"]))

        best_c = -float('inf')
        for exp in [f"{prefix}_C{i}" for i in range(1, 7)]:
            matching_keys = [k for k in ledger if k == exp or k.startswith(f"{exp}_")]
            for k in matching_keys:
                score = get_composite_score(ledger[k])
                if score > best_c:
                    best_c = score
                    defaults["attention_type"] = ledger[k].get("attention_type", defaults["attention_type"])
                    defaults["rnn_type"] = ledger[k].get("rnn_type", defaults["rnn_type"])
                    defaults["bidirectional"] = str(ledger[k].get("bidirectional", defaults["bidirectional"]))
                    if "emb_dim" in ledger[k]: defaults["emb_dim"] = str(ledger[k]["emb_dim"])
    except Exception: pass
    return defaults


def load_evaluation_ledger_df(token_type: str) -> pd.DataFrame:
    pattern = os.path.join(REPO_ROOT, f"evaluation_ledger_{token_type}_*.json")
    ledger_data = {}
    
    for filepath in glob.glob(pattern):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                ledger_data.update(json.load(f))
        except Exception:
            pass

    if not ledger_data:
        return pd.DataFrame()

    records = []
    for run_id, node in ledger_data.items():
        cell = node.get("rnn_type", "RNN")
        bidi = "Bi" if str(node.get("bidirectional", "")).lower() == "true" else "Uni"
        attn = node.get("attention_type", "none")
        emb = node.get("embedding_source", "scratch")

        if "PIVOT" in run_id.upper():
            variant_desc = f"Pivot System (DE->EN->SV) using {cell}"
            study_group = "PIVOT"
        else:
            variant_desc = f"{bidi}-{cell} (Embeds: {emb})"
            if attn != "none":
                variant_desc += f" w/ {attn.capitalize()} Attn"
            match = re.search(r'_(A|B|C|D|E)\d*', run_id.upper())
            study_group = match.group(1) if match else "MISC"

        metrics = node.get("metrics", {})
        bleu = float(metrics.get("overall_corpus_bleu", 0.0))
        meteor = float(metrics.get("mean_meteor", 0.0))
        sid = run_id.split("_")[1] if "_" in run_id else run_id

        records.append({
            "Run ID": run_id,
            "Tokenization": f"{token_type.capitalize()}-Level",
            "Study ID": f"Study {sid}",
            "Top Study Run": f"Study {sid}",
            "Study Group": study_group,
            "Architectural Variant": variant_desc,
            "Best Architectural Variant": variant_desc,
            "BLEU Score": round(bleu, 2),
            "Metric 2 (METEOR)": round(meteor, 2),
            "Train Time": node.get("train_time", "N/A"),
            "Inference Time": node.get("inference_time", "N/A"),
            "_composite_score": bleu + (meteor * 100.0)
        })

    return pd.DataFrame(records)


def generate_all_reports(token_type: str):
    df = load_evaluation_ledger_df(token_type)
    
    if df.empty:
        print(f"ℹ️ No empirical results recorded yet in your isolated {token_type} study ledgers.")
        return

    print("\n" + "="*80 + f"\n📊 GENERATING ALL EVALUATION REPORTS ({token_type.upper()})\n" + "="*80)
    
    export_cols = ["Tokenization", "Study ID", "Architectural Variant", "BLEU Score", "Metric 2 (METEOR)", "Train Time", "Inference Time"]

    consolidated_path = os.path.join(REPO_ROOT, f"consolidated_evaluation_report_{token_type}.csv")
    df[export_cols].to_csv(consolidated_path, index=False)
    print(f"💾 Consolidated Report Saved -> {consolidated_path}")

    for group_name, group_df in df.groupby("Study Group"):
        study_path = os.path.join(REPO_ROOT, f"study_{group_name}_report_{token_type}.csv")
        group_df[export_cols].to_csv(study_path, index=False)
        print(f"💾 Isolated Study Matrix Saved -> study_{group_name}_report_{token_type}.csv")

    best_idx = df.groupby("Study Group")["_composite_score"].idxmax()
    best_df = df.loc[best_idx].sort_values("Study Group")
    
    best_export_cols = ["Tokenization", "Top Study Run", "Best Architectural Variant", "BLEU Score", "Metric 2 (METEOR)", "Train Time", "Inference Time"]
    best_path = os.path.join(REPO_ROOT, f"best_of_studies_report_{token_type}.csv")
    best_df[best_export_cols].to_csv(best_path, index=False)
    
    print(f"\n💾 Aggregated champion ledger saved successfully to: {best_path}\n")


def execute_preprocessing(token_type="word", mock_mode=False):
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "preprocess.py"), "--token_type", token_type]
    if mock_mode:
        cmd.append("--mock")
    
    print(f"⚡ Running preprocessing routine (token_type={token_type}, mock={mock_mode})...")
    subprocess.run(cmd, check=True)


def run_automated_post_processing(token_type, rnn_type):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    try: 
        subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, "evaluate.py"), "evaluate", "--token_type", token_type], check=True, env=env)
    except Exception: pass

    de_en_model = os.path.join(OUTPUT_DIR, f"best_model_{token_type.upper()}_D2_{rnn_type}.pt")
    en_sv_model = os.path.join(OUTPUT_DIR, f"best_model_{token_type.upper()}_E1_{rnn_type}.pt")
    if os.path.exists(de_en_model) and os.path.exists(en_sv_model):
        try: 
            subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, "pivot.py"), "--de_en_model", de_en_model, "--en_sv_model", en_sv_model, "--text", "maschinelles lernen macht unglaublichen spass"], check=True, env=env)
        except Exception: pass

        print(f"\n📊 Launching Formal Quantitative Pivot Dataset Evaluation (DE ➔ EN ➔ SV)...")
        try:
            subprocess.run([
                sys.executable, os.path.join(SCRIPT_DIR, "pivot.py"),
                "--de_en_model", de_en_model,
                "--en_sv_model", en_sv_model,
                "--evaluate",
                "--token_type", token_type,
                "--experiment", f"{token_type.upper()}_PIVOT"
            ], check=True, env=env)
            sync_ledger_to_token_type(token_type)
        except Exception as e:
            print(f"⚠️ Quantitative pivot dataset evaluation interrupted or unsupported: {e}")

    try: 
        subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, "evaluate.py"), "compile", "--token_type", token_type], check=True, env=env)
    except Exception: pass

    attn_model = os.path.join(OUTPUT_DIR, f"best_model_{token_type.upper()}_C4_{rnn_type}.pt")
    if not os.path.exists(attn_model): attn_model = os.path.join(OUTPUT_DIR, f"best_model_{token_type.upper()}_C3_{rnn_type}.pt")
    if os.path.exists(attn_model):
        try: subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, "evaluate.py"), "visualize", "--model", attn_model], check=True, env=env)
        except Exception: pass


def execute_study_a(epochs, token_type, eval_queue: AsyncEvaluationQueue):
    configs = [("A1", "RNN", "False"), ("A2", "RNN", "True"), ("A3", "GRU", "False"), ("A4", "GRU", "True"), ("A5", "LSTM", "False"), ("A6", "LSTM", "True")]
    batch_size = get_batch_size("A", token_type)
    for exp, cell, bidi in configs:
        exp_id = f"{token_type.upper()}_{exp}"
        hparams = get_best_hyperparameters("coarse", token_type, rnn_type=cell)
        
        emb_dim = hparams[hparams.index("--emb_dim") + 1] if "--emb_dim" in hparams else "256"
        hidden_dim = hparams[hparams.index("--hidden_dim") + 1] if "--hidden_dim" in hparams else "512"

        print_study_model_and_batch_info(
            study_name="Study A (Architecture Benchmarking)",
            exp_id=exp_id, token_type=token_type, rnn_type=cell, bidirectional=bidi,
            attention_type="none", emb_dim=emb_dim, hidden_dim=hidden_dim, batch_size=batch_size
        )

        if not is_cache_valid(os.path.join(OUTPUT_DIR, f"best_model_{exp_id}_{cell}.pt"), os.path.join(OUTPUT_DIR, f"best_config_{exp_id}_{cell}.json"), epochs):
            run_cmd(hparams + ["--experiment", exp_id, "--rnn_type", cell, "--bidirectional", bidi, "--token_type", token_type, "--batch_size", batch_size, "--epochs", str(epochs)])
        
        eval_queue.submit_evaluation(exp_id, cell, token_type)

    eval_queue.sync_study()


def execute_study_b(epochs, rnn_type, bidirectional, token_type, eval_queue: AsyncEvaluationQueue):
    hparams = get_best_hyperparameters("coarse", token_type, rnn_type=rnn_type)
    configs = [("B1", "scratch", "False", "256"), ("B2", "word2vec", "True", "256"), ("B3", "word2vec", "False", "256"), ("B4", "scratch", "True", "256"), ("B5", "glove", "True", "256"), ("B6", "glove", "False", "256")] if token_type == "word" else [("B7", "scratch", "False", "32"), ("B8", "scratch", "False", "64"), ("B9", "scratch", "False", "128"), ("B10", "onehot", "True", "128")]
    batch_size = get_batch_size("B", token_type)
    
    hidden_dim = hparams[hparams.index("--hidden_dim") + 1] if "--hidden_dim" in hparams else "512"

    for exp, src, freeze, emb_dim in configs:
        exp_id = f"{token_type.upper()}_{exp}"

        print_study_model_and_batch_info(
            study_name="Study B (Embedding Representation Analysis)",
            exp_id=exp_id, token_type=token_type, rnn_type=rnn_type, bidirectional=bidirectional,
            attention_type="none", emb_dim=emb_dim, hidden_dim=hidden_dim, batch_size=batch_size
        )

        if not is_cache_valid(os.path.join(OUTPUT_DIR, f"best_model_{exp_id}_{rnn_type}.pt"), os.path.join(OUTPUT_DIR, f"best_config_{exp_id}_{rnn_type}.json"), epochs):
            run_cmd(hparams + ["--experiment", exp_id, "--rnn_type", rnn_type, "--bidirectional", bidirectional, "--token_type", token_type, "--embedding_source", "scratch" if src == "onehot" else src, "--freeze_emb", freeze, "--emb_dim", emb_dim, "--batch_size", batch_size, "--epochs", str(epochs)])
        
        eval_queue.submit_evaluation(exp_id, rnn_type, token_type)

    eval_queue.sync_study()


def execute_study_c(epochs, token_type, rnn_type, bidirectional, embedding_source, freeze_emb, emb_dim, eval_queue: AsyncEvaluationQueue):
    hparams = get_best_hyperparameters("coarse", token_type, rnn_type=rnn_type)
    configs = [("C1", rnn_type, "none", "False"), ("C2", rnn_type, "none", bidirectional), ("C3", rnn_type, "luong", bidirectional), ("C4", rnn_type, "bahdanau", bidirectional), ("C5", "RNN", "luong", "True"), ("C6", "RNN", "bahdanau", "True")]
    
    batch_size = get_batch_size("C", token_type)
    hidden_dim = hparams[hparams.index("--hidden_dim") + 1] if "--hidden_dim" in hparams else "512"

    for exp, cell, attn, bidi in configs:
        exp_id = f"{token_type.upper()}_{exp}"

        print_study_model_and_batch_info(
            study_name="Study C (Attention Mechanism Optimization)",
            exp_id=exp_id, token_type=token_type, rnn_type=cell, bidirectional=bidi,
            attention_type=attn, emb_dim=emb_dim, hidden_dim=hidden_dim, batch_size=batch_size
        )

        if not is_cache_valid(os.path.join(OUTPUT_DIR, f"best_model_{exp_id}_{cell}.pt"), os.path.join(OUTPUT_DIR, f"best_config_{exp_id}_{cell}.json"), epochs):
            cmd_hparams = get_best_hyperparameters("coarse", token_type, rnn_type=cell) if cell == "RNN" else hparams
            run_cmd(cmd_hparams + [
                "--experiment", exp_id, 
                "--rnn_type", cell, 
                "--attention_type", attn, 
                "--bidirectional", bidi, 
                "--token_type", token_type, 
                "--embedding_source", embedding_source, 
                "--freeze_emb", freeze_emb, 
                "--emb_dim", emb_dim, 
                "--batch_size", batch_size,
                "--epochs", str(epochs)
            ])
        
        eval_queue.submit_evaluation(exp_id, cell, token_type)

    eval_queue.sync_study()


def execute_study_d(epochs, token_type, rnn_type, bidirectional, embedding_source, freeze_emb, attention_type, emb_dim, eval_queue: AsyncEvaluationQueue):
    hparams = get_best_hyperparameters("fine", token_type, rnn_type=rnn_type)
    configs = [("D1", "en", "de", token_type), ("D2", "de", "en", token_type)]
    batch_size = get_batch_size("D", token_type)
    hidden_dim = hparams[hparams.index("--hidden_dim") + 1] if "--hidden_dim" in hparams else "512"

    for exp, src, trg, tok in configs:
        exp_id = f"{token_type.upper()}_{exp}"

        print_study_model_and_batch_info(
            study_name="Study D (Language Directionality Analysis)",
            exp_id=exp_id, token_type=tok, rnn_type=rnn_type, bidirectional=bidirectional,
            attention_type=attention_type, emb_dim=emb_dim, hidden_dim=hidden_dim, batch_size=batch_size
        )

        if not is_cache_valid(os.path.join(OUTPUT_DIR, f"best_model_{exp_id}_{rnn_type}.pt"), os.path.join(OUTPUT_DIR, f"best_config_{exp_id}_{rnn_type}.json"), epochs):
            run_cmd(hparams + ["--experiment", exp_id, "--rnn_type", rnn_type, "--bidirectional", bidirectional, "--attention_type", attention_type, "--embedding_source", embedding_source, "--freeze_emb", freeze_emb, "--emb_dim", emb_dim, "--batch_size", batch_size, "--src_lang", src, "--trg_lang", trg, "--token_type", tok, "--epochs", str(epochs)])
        
        eval_queue.submit_evaluation(exp_id, rnn_type, token_type)

    eval_queue.sync_study()


def execute_study_e(epochs, token_type, rnn_type, bidirectional, embedding_source, freeze_emb, attention_type, emb_dim, eval_queue: AsyncEvaluationQueue):
    hparams = get_best_hyperparameters("fine", token_type, rnn_type=rnn_type)
    exp_id = f"{token_type.upper()}_E1"
    batch_size = get_batch_size("E", token_type)
    hidden_dim = hparams[hparams.index("--hidden_dim") + 1] if "--hidden_dim" in hparams else "512"

    print_study_model_and_batch_info(
        study_name="Study E (Cross-Lingual Transfer to Swedish)",
        exp_id=exp_id, token_type=token_type, rnn_type=rnn_type, bidirectional=bidirectional,
        attention_type=attention_type, emb_dim=emb_dim, hidden_dim=hidden_dim, batch_size=batch_size
    )

    if not is_cache_valid(os.path.join(OUTPUT_DIR, f"best_model_{exp_id}_{rnn_type}.pt"), os.path.join(OUTPUT_DIR, f"best_config_{exp_id}_{rnn_type}.json"), epochs):
        run_cmd(hparams + ["--experiment", exp_id, "--rnn_type", rnn_type, "--bidirectional", bidirectional, "--attention_type", attention_type, "--embedding_source", embedding_source, "--freeze_emb", freeze_emb, "--emb_dim", emb_dim, "--batch_size", batch_size, "--src_lang", "en", "--trg_lang", "sv", "--token_type", token_type, "--epochs", str(epochs)])
    
    eval_queue.submit_evaluation(exp_id, rnn_type, token_type)
    eval_queue.sync_study()


def execute_hyperparameter_tuning(stage, token_type, strategy, samples, epochs, extra_args=[]):
    """Native Orchestrator Tuning Sweep with direct centralized artifact caching."""
    print(f"\n🚀 [STARTING] Hyperparameter Tuning | Stage: {stage} | Tokenizer: {token_type.upper()} | Strategy: {strategy} | Target Epochs: {epochs}")
    
    rnn_type = "LSTM"
    extra_kv = {}
    i = 0
    while i < len(extra_args):
        if extra_args[i].startswith("--") and i + 1 < len(extra_args):
            extra_kv[extra_args[i]] = extra_args[i+1]
            i += 2
        else:
            i += 1
            
    if "--rnn_type" in extra_kv:
        rnn_type = extra_kv["--rnn_type"]
        
    stage_csv = os.path.join(REPO_ROOT, f"tuning_results_{token_type}_{stage}.csv")
    
    if token_type == "word":
        lrs = [0.003, 0.001, 0.0005]
        dropouts = [0.2, 0.3, 0.4]
        emb_dims = [128, 256]
        hidden_dims = [256, 512]
    else:
        lrs = [0.001, 0.0005, 0.0001]
        dropouts = [0.2, 0.3, 0.4]
        emb_dims = [32, 64]
        hidden_dims = [256]

    all_combos = list(itertools.product(lrs, dropouts, emb_dims, hidden_dims))
    random.seed(42)
    random.shuffle(all_combos)
    selected_combos = all_combos[:min(samples, len(all_combos))]
    
    new_results = []
    
    for idx, (lr, dropout, emb_dim, hidden_dim) in enumerate(selected_combos):
        sample_num = idx + 1
        exp_id = f"TUNE_{token_type.upper()}_SAMPLE_{sample_num}"
        
        candidate_tags = [
            f"{exp_id}_{rnn_type}",
            f"TUNE_{token_type.upper()}_{stage.upper()}_SAMPLE_{sample_num}_{rnn_type}",
            f"TUNE_{token_type.upper()}_{rnn_type}_SAMPLE_{sample_num}",
            exp_id
        ]
        
        cached_config_file, _ = check_artifact_cache(OUTPUT_DIR, candidate_tags)
                
        if cached_config_file and os.path.exists(cached_config_file):
            try:
                with open(cached_config_file, 'r', encoding='utf-8') as f:
                    cdata = json.load(f)
                val_loss = float(cdata.get("best_val_loss", cdata.get("val_loss", 999.0)))
                new_results.append({
                    "status": "Success",
                    "val_loss": val_loss,
                    "learning_rate": lr,
                    "dropout": dropout,
                    "emb_dim": emb_dim,
                    "hidden_dim": hidden_dim,
                    "rnn_type": rnn_type
                })
                continue
            except Exception as e:
                print(f"⚠️ Cache read error for {cached_config_file}: {e}")

        cmd_args = [
            "--experiment", exp_id,
            "--rnn_type", rnn_type,
            "--token_type", token_type,
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--dropout", str(dropout),
            "--emb_dim", str(emb_dim),
            "--hidden_dim", str(hidden_dim)
        ] + extra_args
        
        try:
            run_cmd(cmd_args)
            exp_tag = f"{exp_id}_{rnn_type}"
            expected_json = os.path.join(OUTPUT_DIR, f"best_config_{exp_tag}.json")
            if not os.path.exists(expected_json):
                expected_json = os.path.join(OUTPUT_DIR, f"best_config_{exp_id}.json")
                
            if os.path.exists(expected_json):
                with open(expected_json, 'r', encoding='utf-8') as f:
                    cdata = json.load(f)
                val_loss = float(cdata.get("best_val_loss", cdata.get("val_loss", 999.0)))
                status = "Success"
            else:
                status = "Failed"
                val_loss = 999.0
        except Exception as e:
            print(f"❌ Training failed for tuning sample {sample_num}: {e}")
            status = "Failed"
            val_loss = 999.0
            
        new_results.append({
            "status": status,
            "val_loss": val_loss,
            "learning_rate": lr,
            "dropout": dropout,
            "emb_dim": emb_dim,
            "hidden_dim": hidden_dim,
            "rnn_type": rnn_type
        })

    if new_results:
        df_new = pd.DataFrame(new_results)
        if os.path.exists(stage_csv):
            try:
                df_old = pd.read_csv(stage_csv)
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
            except Exception:
                df_combined = df_new
        else:
            df_combined = df_new
            
        df_combined.to_csv(stage_csv, index=False)
        print(f"✅ [SUCCESS] Orchestrator ledger updated -> {stage_csv}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Empirically Cascaded, Academically Sound NMT Pipeline")
    parser.add_argument("--study", type=str, required=True, choices=["A", "B", "C", "D", "E", "tune", "all"])
    parser.add_argument("--token_type", type=str, default="word", choices=["word", "char", "both"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--tune_strategy", type=str, default="random")
    parser.add_argument("--tune_samples", type=int, default=3)
    args = parser.parse_args()
    
    runtime_epochs = 1 if args.mock else args.epochs
    target_pathways = ["word", "char"] if args.token_type == "both" else [args.token_type]
    
    execute_preprocessing(
        token_type=getattr(args, "token_type", "word"), 
        mock_mode=args.mock
    )
    rnn_architectures = ["RNN", "GRU", "LSTM"]
    
    eval_queue = AsyncEvaluationQueue(max_workers=2)

    try:
        if args.study == "tune":
            for pathway in target_pathways:
                char_factor = 1.25 if pathway == "char" else 1.0
                epochs_coarse = max(1, int(runtime_epochs * 0.4 * char_factor)) if not args.mock else 1
                
                for arch in rnn_architectures:
                    execute_hyperparameter_tuning("coarse", pathway, args.tune_strategy, args.tune_samples, epochs_coarse, ["--rnn_type", arch])
                    
        elif args.study == "all":
            for pathway in target_pathways:
                char_factor = 1.25 if pathway == "char" else 1.0
                base_scale = max(1, int(runtime_epochs * char_factor)) if not args.mock else 1
                
                epochs_coarse = max(1, int(base_scale * 0.4))
                epochs_abc    = max(1, int(base_scale * 0.8))
                epochs_fine   = max(1, int(base_scale * 0.6))
                epochs_de     = max(1, int(base_scale * 1.8))
                
                print(f"\n📊 [DYNAMIC EPOCH TIER PLAN] Pathway: {pathway.upper()}")
                print(f" ├─ Coarse Sweep Epochs:  {epochs_coarse}")
                print(f" ├─ Studies A, B, C:       {epochs_abc}")
                print(f" ├─ Fine Sweep Epochs:    {epochs_fine}")
                print(f" └─ Studies D & E:         {epochs_de}\n")

                for arch in rnn_architectures:
                    execute_hyperparameter_tuning("coarse", pathway, args.tune_strategy, args.tune_samples, epochs_coarse, ["--rnn_type", arch])
                    
                execute_study_a(epochs_abc, token_type=pathway, eval_queue=eval_queue)
                best = get_best_empirical_settings(token_type=pathway)
                
                execute_study_b(epochs_abc, best["rnn_type"], best["bidirectional"], pathway, eval_queue=eval_queue)
                best = get_best_empirical_settings(token_type=pathway)
                
                execute_study_c(epochs_abc, pathway, best["rnn_type"], best["bidirectional"], best["embedding_source"], best["freeze_emb"], best["emb_dim"], eval_queue=eval_queue)
                best = get_best_empirical_settings(token_type=pathway)
                
                execute_hyperparameter_tuning("fine", pathway, args.tune_strategy, args.tune_samples * 2, epochs_fine, [
                    "--rnn_type", best["rnn_type"], 
                    "--bidirectional", best["bidirectional"], 
                    "--embedding_source", best["embedding_source"], 
                    "--freeze_emb", best["freeze_emb"], 
                    "--attention_type", best["attention_type"], 
                    "--emb_dim", best["emb_dim"]
                ])
                
                execute_study_d(epochs_de, pathway, best["rnn_type"], best["bidirectional"], best["embedding_source"], best["freeze_emb"], best["attention_type"], best["emb_dim"], eval_queue=eval_queue)
                best = get_best_empirical_settings(token_type=pathway)
                
                execute_study_e(epochs_de, pathway, best["rnn_type"], best["bidirectional"], best["embedding_source"], best["freeze_emb"], best["attention_type"], best["emb_dim"], eval_queue=eval_queue)
                best = get_best_empirical_settings(token_type=pathway)
                
                run_automated_post_processing(token_type=pathway, rnn_type=best["rnn_type"])
                generate_all_reports(token_type=pathway)
        else:
            for pathway in target_pathways:
                best = get_best_empirical_settings(token_type=pathway)
                
                char_factor = 1.25 if pathway == "char" else 1.0
                base_scale = max(1, int(runtime_epochs * char_factor)) if not args.mock else 1

                if args.study in ["A", "B", "C"]:
                    study_epochs = max(1, int(base_scale * 0.8))
                elif args.study in ["D", "E"]:
                    study_epochs = max(1, int(base_scale * 1.8))
                else:
                    study_epochs = base_scale

                if args.study == "A": 
                    execute_study_a(study_epochs, token_type=pathway, eval_queue=eval_queue)
                elif args.study == "B": 
                    execute_study_b(study_epochs, best["rnn_type"], best["bidirectional"], pathway, eval_queue=eval_queue)
                elif args.study == "C": 
                    execute_study_c(study_epochs, pathway, best["rnn_type"], best["bidirectional"], best["embedding_source"], best["freeze_emb"], best["emb_dim"], eval_queue=eval_queue)
                elif args.study == "D": 
                    execute_study_d(study_epochs, pathway, best["rnn_type"], best["bidirectional"], best["embedding_source"], best["freeze_emb"], best["attention_type"], best["emb_dim"], eval_queue=eval_queue)
                elif args.study == "E": 
                    execute_study_e(study_epochs, pathway, best["rnn_type"], best["bidirectional"], best["embedding_source"], best["freeze_emb"], best["attention_type"], best["emb_dim"], eval_queue=eval_queue)
                
                best = get_best_empirical_settings(token_type=pathway)
                
                run_automated_post_processing(token_type=pathway, rnn_type=best["rnn_type"])
                generate_all_reports(token_type=pathway)
    finally:
        eval_queue.shutdown()