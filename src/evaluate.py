import os
import sys
import json
import re
import csv
import time
import glob
import multiprocessing
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import get_dataloader, SOS_IDX, EOS_IDX, PAD_IDX
from models import Encoder, Decoder, Seq2Seq

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "results")

def _worker_meteor(pair):
    ref, hyp = pair
    return meteor_score(ref, hyp)

def translate_sentence(model, src_tensor, trg_vocab, device, max_len=50):
    model.eval()
    tokens = []
    
    with torch.no_grad():
        # Correctly unpack 2 outputs from encoder
        encoder_outputs, hidden = model.encoder(src_tensor)
        hidden = model._bridge_hidden(hidden)
        
        current_token = torch.tensor([SOS_IDX], dtype=torch.long, device=device)
        
        for _ in range(max_len):
            # Forward step returns (prediction, hidden)
            prediction, hidden = model.decoder.forward_step(current_token, hidden, encoder_outputs)
            best_guess = prediction.argmax(dim=1).item()
            
            if best_guess == EOS_IDX: 
                break
            if best_guess != PAD_IDX:
                tokens.append(trg_vocab.itos.get(best_guess, "<unk>"))
                
            current_token = torch.tensor([best_guess], dtype=torch.long, device=device)
            
    return tokens


def translate_batch(model, src_tensor, trg_vocab, device, max_len=50):
    model.eval()
    batch_size = src_tensor.size(0)
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        hidden = model._bridge_hidden(hidden)
        
        current_tokens = torch.full((batch_size,), SOS_IDX, dtype=torch.long, device=device)
        outputs = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        
        for t in range(max_len):
            prediction, hidden = model.decoder.forward_step(current_tokens, hidden, encoder_outputs)
            best_guess = prediction.argmax(dim=1)
            outputs[:, t] = best_guess
            current_tokens = best_guess
        
    outputs_cpu = outputs.cpu().tolist()
    translated_sentences = []
    for i in range(batch_size):
        tokens = []
        for idx in outputs_cpu[i]:
            if idx == EOS_IDX: 
                break
            if idx != PAD_IDX:
                tokens.append(trg_vocab.itos.get(idx, "<unk>"))
        translated_sentences.append(tokens)
    return translated_sentences

def run_evaluation(checkpoint_path, test_csv=None, sample_size=None, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⌛ Loading checkpoint: '{os.path.basename(checkpoint_path)}'")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    state_dict = checkpoint['model_state_dict']
    src_vocab = checkpoint.get('src_vocab')
    trg_vocab = checkpoint.get('trg_vocab')

    src_lang, trg_lang = config.get('src_lang', 'en'), config.get('trg_lang', 'de')
    token_type, experiment_id = config.get('token_type', 'word'), config.get('experiment', 'Unknown')

    # Differentiate max generation length between character and word levels
    max_len = 250 if token_type == "char" else 50

    if not test_csv:
        test_csv = os.path.join(ROOT_DIR, "data", "processed", f"test_{src_lang}_{trg_lang}.csv")
        if not os.path.exists(test_csv):
            # Fallback for legacy files
            legacy_file = "test_sv.csv" if ("sv" in (src_lang, trg_lang)) else "test.csv"
            test_csv = os.path.join(ROOT_DIR, "data", "processed", legacy_file)
            
    if not os.path.exists(test_csv): 
        return
        
    test_loader, _, _ = get_dataloader(test_csv, batch_size=config.get('batch_size', 2048), shuffle=False, src_vocab=src_vocab, trg_vocab=trg_vocab, src_lang=src_lang, trg_lang=trg_lang, token_type=token_type)
    
    # Dataset Subsampling Logic
    if sample_size is not None:
        total_len = len(test_loader.dataset)
        if isinstance(sample_size, float) and 0.0 < sample_size <= 1.0:
            target_count = int(total_len * sample_size)
        else:
            target_count = min(int(sample_size), total_len)
        
        if target_count < total_len:
            print(f"🎲 Subsampling test dataset: evaluating on {target_count}/{total_len} samples (seed={seed})")
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(total_len, generator=generator)[:target_count].tolist()
            subset_dataset = Subset(test_loader.dataset, indices)
            
            test_loader = DataLoader(
                subset_dataset,
                batch_size=test_loader.batch_size,
                shuffle=False,
                collate_fn=getattr(test_loader, 'collate_fn', None),
                num_workers=getattr(test_loader, 'num_workers', 0)
            )

    is_bidi = config.get('bidirectional', True)
    enc_hidden_dim = config['hidden_dim'] * (2 if is_bidi else 1)

    enc_rnn_in_dim = state_dict['encoder.project.weight'].shape[0] if 'encoder.project.weight' in state_dict else state_dict['encoder.embedding.weight'].shape[1]
    dec_rnn_in_dim = state_dict['decoder.project.weight'].shape[0] if 'decoder.project.weight' in state_dict else state_dict['decoder.embedding.weight'].shape[1]

    enc = Encoder(state_dict['encoder.embedding.weight'].shape[0], enc_rnn_in_dim, config['hidden_dim'], config.get('n_layers', 2), config.get('dropout', 0.3), rnn_type=config['rnn_type'], bidirectional=is_bidi)
    dec = Decoder(state_dict['decoder.embedding.weight'].shape[0], dec_rnn_in_dim, enc_hidden_dim, config['hidden_dim'], config.get('n_layers', 2), config.get('dropout', 0.3), rnn_type=config['rnn_type'], attention_type=config.get('attention_type', 'none'))
    
    if 'encoder.project.weight' in state_dict: 
        enc.embedding = torch.nn.Embedding(state_dict['encoder.embedding.weight'].shape[0], state_dict['encoder.embedding.weight'].shape[1])
        enc.project = torch.nn.Linear(state_dict['encoder.project.weight'].shape[1], state_dict['encoder.project.weight'].shape[0])
    if 'decoder.project.weight' in state_dict: 
        dec.embedding = torch.nn.Embedding(state_dict['decoder.embedding.weight'].shape[0], state_dict['decoder.embedding.weight'].shape[1])
        dec.project = torch.nn.Linear(state_dict['decoder.project.weight'].shape[1], state_dict['decoder.project.weight'].shape[0])
    if 'decoder.fc_out.weight' in state_dict: 
        dec.fc_out = torch.nn.Linear(state_dict['decoder.fc_out.weight'].shape[1], state_dict['decoder.fc_out.weight'].shape[0])
    
    model = Seq2Seq(enc, dec, device).to(device)
    model.load_state_dict(state_dict)
    
    if hasattr(torch, "compile"):
        model = torch.compile(model)
    
    references, hypotheses = [], []
    meta_info = []
    
    start_time = time.time()
    for src, trg in test_loader:
        src = src.to(device, non_blocking=True)
        # Pass the token-type specific max_len to translation batch
        batch_hyps = translate_batch(model, src, trg_vocab, device, max_len=max_len)
        hypotheses.extend(batch_hyps)
        
        trg_list = trg.tolist()
        src_list = src.tolist()
        
        for i in range(len(trg_list)):
            ref_tokens = [trg_vocab.itos[idx] for idx in trg_list[i] if idx not in [PAD_IDX, SOS_IDX, EOS_IDX]]
            references.append([ref_tokens])
            
            src_tokens = [src_vocab.itos[idx] for idx in src_list[i] if idx not in [PAD_IDX, SOS_IDX, EOS_IDX]]
            meta_info.append(len(src_tokens))
                
    avg_inference_ms = ((time.time() - start_time) / max(1, len(hypotheses))) * 1000
    
    print("📊 Computing METEOR scores in parallel across CPU cores...")
    meteor_pairs = list(zip(references, hypotheses))
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        meteor_scores = pool.map(_worker_meteor, meteor_pairs)
        
    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1) * 100
    mean_meteor = np.mean(meteor_scores) if meteor_scores else 0.0
    
    print(f"✨ Score Summary [{experiment_id}] -> BLEU: {bleu_score:.2f} | METEOR: {mean_meteor:.4f}")
    
    buckets = {
        "Short (1-10 tokens)": {"refs": [], "hyps": [], "meteors": []},
        "Medium (11-20 tokens)": {"refs": [], "hyps": [], "meteors": []},
        "Long (21-30 tokens)": {"refs": [], "hyps": [], "meteors": []},
        "Very Long (31+ tokens)": {"refs": [], "hyps": [], "meteors": []}
    }
    
    for idx, src_len in enumerate(meta_info):
        if src_len <= 10:
            b_key = "Short (1-10 tokens)"
        elif src_len <= 20:
            b_key = "Medium (11-20 tokens)"
        elif src_len <= 30:
            b_key = "Long (21-30 tokens)"
        else:
            b_key = "Very Long (31+ tokens)"
            
        buckets[b_key]["refs"].append(references[idx])
        buckets[b_key]["hyps"].append(hypotheses[idx])
        buckets[b_key]["meteors"].append(meteor_scores[idx])

    bucket_analysis_results = {}
    for b_name, b_data in buckets.items():
        if len(b_data["hyps"]) > 0:
            b_bleu = corpus_bleu(b_data["refs"], b_data["hyps"], smoothing_function=SmoothingFunction().method1) * 100
            b_meteor = np.mean(b_data["meteors"])
            bucket_analysis_results[b_name] = {
                "sample_count": len(b_data["hyps"]),
                "bleu": round(b_bleu, 2),
                "meteor": round(b_meteor, 4)
            }
        else:
            bucket_analysis_results[b_name] = {"sample_count": 0, "bleu": 0.0, "meteor": 0.0}

    ledger_path = os.path.join(ROOT_DIR, f"evaluation_ledger_{token_type}.json")
    os.makedirs(ROOT_DIR, exist_ok=True)
    ledger_data = {}
    if os.path.exists(ledger_path):
        try:
            with open(ledger_path, 'r') as f: 
                ledger_data = json.load(f)
        except Exception: 
            pass
            
    ledger_data[experiment_id] = {
        "rnn_type": config['rnn_type'], "bidirectional": is_bidi,
        "embedding_source": config.get("embedding_source", "scratch"),
        "freeze_emb": config.get("freeze_emb", False), "attention_type": config.get("attention_type", "none"),
        "train_time": config.get("train_time", "N/A"), "inference_time": f"{avg_inference_ms:.1f}ms / sent",
        "metrics": {"overall_corpus_bleu": bleu_score, "mean_meteor": mean_meteor},
        "length_bucket_analysis": bucket_analysis_results
    }
    with open(ledger_path, 'w') as f: 
        json.dump(ledger_data, f, indent=4)

def evaluate_all_local_models(token_type=None, sample_size=None, seed=42):
    search_dir = OUTPUT_DIR if os.path.exists(OUTPUT_DIR) else '.'
    for file in sorted(os.listdir(search_dir)):
        if file.startswith("best_model_") and file.endswith(".pt"):
            if token_type and f"_{token_type.upper()}_" not in file.upper():
                continue
            try: 
                run_evaluation(os.path.join(search_dir, file), sample_size=sample_size, seed=seed)
            except Exception as e: 
                print(f"⚠️ Skipping corrupt layout {file}: {e}")

def visualize_sample_attention(model_path, test_csv, sample_index=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_name = os.path.basename(model_path).replace(".pt", "")
    fallback_csv_path = os.path.join(ROOT_DIR, f"attention_data_fallback_{base_name}.csv")
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = checkpoint['config']
        state_dict = checkpoint['model_state_dict']
        src_vocab = checkpoint['src_vocab']
        trg_vocab = checkpoint['trg_vocab']

        if config.get('attention_type', 'none') == 'none':
            with open(fallback_csv_path, 'w', newline='') as f:
                csv.writer(f).writerows([["Model Path", "Attention Type"], [model_path, "None"]])
            return
            
        test_loader, _, _ = get_dataloader(test_csv, batch_size=1, shuffle=False, src_vocab=src_vocab, trg_vocab=trg_vocab, src_lang=config['src_lang'], trg_lang=config['trg_lang'], token_type=config.get('token_type', 'word'))
        
        is_bidi = config.get('bidirectional', True)
        enc_hidden_dim = config['hidden_dim'] * (2 if is_bidi else 1)

        enc_rnn_in_dim = state_dict['encoder.project.weight'].shape[0] if 'encoder.project.weight' in state_dict else state_dict['encoder.embedding.weight'].shape[1]
        dec_rnn_in_dim = state_dict['decoder.project.weight'].shape[0] if 'decoder.project.weight' in state_dict else state_dict['decoder.embedding.weight'].shape[1]

        enc = Encoder(state_dict['encoder.embedding.weight'].shape[0], enc_rnn_in_dim, config['hidden_dim'], config.get('n_layers', 2), config.get('dropout', 0.3), rnn_type=config['rnn_type'], bidirectional=is_bidi)
        dec = Decoder(state_dict['decoder.embedding.weight'].shape[0], dec_rnn_in_dim, enc_hidden_dim, config['hidden_dim'], config.get('n_layers', 2), config.get('dropout', 0.3), rnn_type=config['rnn_type'], attention_type=config.get('attention_type', 'none'))
        
        if 'encoder.project.weight' in state_dict: 
            enc.embedding = torch.nn.Embedding(state_dict['encoder.embedding.weight'].shape[0], state_dict['encoder.embedding.weight'].shape[1])
            enc.project = torch.nn.Linear(state_dict['encoder.project.weight'].shape[1], state_dict['encoder.project.weight'].shape[0])
        if 'decoder.project.weight' in state_dict: 
            dec.embedding = torch.nn.Embedding(state_dict['decoder.embedding.weight'].shape[0], state_dict['decoder.embedding.weight'].shape[1])
            dec.project = torch.nn.Linear(state_dict['decoder.project.weight'].shape[1], state_dict['decoder.project.weight'].shape[0])
        if 'decoder.fc_out.weight' in state_dict: 
            dec.fc_out = torch.nn.Linear(state_dict['decoder.fc_out.weight'].shape[1], state_dict['decoder.fc_out.weight'].shape[0])

        model = Seq2Seq(enc, dec, device).to(device)
        model.load_state_dict(state_dict)

        if hasattr(torch, "compile"):
            model = torch.compile(model)

        for idx, (src, _) in enumerate(test_loader):
            if idx == sample_index:
                src = src.to(device)
                outputs, attentions = [SOS_IDX], []
                
                with torch.no_grad(): 
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        encoder_outputs, hidden, cell = model.encoder(src)
                        previous_word = torch.tensor([SOS_IDX], dtype=torch.long, device=device)
                        
                        max_viz_len = 250 if config.get('token_type', 'word') == "char" else 50
                        for _ in range(max_viz_len):
                            prediction, hidden, cell, attn_weights = model.decoder(previous_word, hidden, cell, encoder_outputs)
                            best_guess = prediction.argmax(1).item()
                            outputs.append(best_guess)
                            attentions.append(attn_weights.squeeze(0).squeeze(0).cpu().numpy())
                            if best_guess == EOS_IDX: 
                                break
                            previous_word.fill_(best_guess)
                
                src_tokens = [src_vocab.itos[i.item()] for i in src.squeeze() if i.item() != PAD_IDX]
                trg_tokens = [trg_vocab.itos[i] for i in outputs[1:]]
                attention_matrix = np.stack(attentions, axis=0)[:len(trg_tokens), :len(src_tokens)]
                
                with open(fallback_csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Target\\Source"] + src_tokens)
                    for t_idx, t_tok in enumerate(trg_tokens): 
                        writer.writerow([t_tok] + list(attention_matrix[t_idx]))
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(attention_matrix, xticklabels=src_tokens, yticklabels=trg_tokens, cmap='viridis')
                plt.tight_layout()
                
                token_tag = config.get('token_type', 'unknown')
                plt.savefig(os.path.join(ROOT_DIR, f"attention_map_{token_tag}_{config['rnn_type']}_{config['attention_type']}.png"), dpi=300)
                plt.close()
                break
    except Exception as e:
        with open(fallback_csv_path, 'w', newline='') as f: 
            csv.writer(f).writerow(["Error", str(e)])

def extract_study_id(filename):
    match = re.search(r'best_config_(.+?)_[A-Z]+\.json$', filename)
    if match:
        return match.group(1).upper()
    return filename.replace("best_config_", "").replace(".json", "").upper()

def generate_csv_file(filepath, data_dict):
    headers = ["Study", "Model Config Tag", "Hyperparameters (LR / Dropout)", "Best Val Loss", "BLEU (↑)", "METEOR (↑)"]
    try:
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for study_id in sorted([k for k in data_dict.keys() if k != "length_bucket_analysis"]):
                cfg = data_dict[study_id]
                config_desc = f"{cfg.get('rnn_type', 'Seq2Seq')}"
                if study_id.startswith("B"): 
                    config_desc += f" + {cfg.get('embedding_source','Default')} Embeds"
                elif study_id.startswith("C"): 
                    config_desc += f" + {cfg.get('attention_type','None')} Attention"
                elif study_id.startswith("D"): 
                    config_desc += " + Inverse Translation"
                elif study_id.startswith("E"): 
                    config_desc += " + Pivot Bridge"
                
                bleu_val = cfg.get('best_bleu', 'N/A')
                meteor_val = cfg.get('best_meteor', 'N/A')
                
                if isinstance(bleu_val, float): 
                    bleu_val = f"{bleu_val:.2f}"
                if isinstance(meteor_val, float): 
                    meteor_val = f"{meteor_val:.4f}"
                
                writer.writerow([
                    f"Study {study_id}", 
                    config_desc, 
                    f"LR: {cfg.get('lr','N/A')}, Drop: {cfg.get('dropout','N/A')}", 
                    f"{cfg.get('best_val_loss','N/A')}", 
                    bleu_val, 
                    meteor_val
                ])
    except Exception: 
        pass

def run_compile_results():
    search_dir = OUTPUT_DIR if os.path.exists(OUTPUT_DIR) else '.'
    master_ledger = {"metadata": {"status": "Complete"}, "word_level": {}, "character_level": {}}
    
    for file in sorted(os.listdir(search_dir)):
        if file.startswith("best_config_") and file.endswith(".json"):
            try:
                with open(os.path.join(search_dir, file), 'r') as f: 
                    config_data = json.load(f)
                study_letter = extract_study_id(file)
                category = "word_level" if "WORD" in file.upper() else "character_level"
                master_ledger[category][study_letter] = config_data
            except Exception: 
                pass

    for category, token_key in [("word_level", "word"), ("character_level", "char")]:
        master_ledger[category]["length_bucket_analysis"] = {}
        for folder in [search_dir, ROOT_DIR]:
            pattern = os.path.join(folder, f"evaluation_ledger_{token_key}_*.json")
            for filepath in glob.glob(pattern):
                try:
                    with open(filepath, 'r') as f:
                        ledger_data = json.load(f)
                        master_ledger[category]["length_bucket_analysis"].update(ledger_data)
                except Exception: 
                    pass
        
        for study_id, study_cfg in list(master_ledger[category].items()):
            if study_id == "length_bucket_analysis": 
                continue
            prefix = "WORD_" if category == "word_level" else "CHAR_"
            metrics = master_ledger[category]["length_bucket_analysis"].get(f"{prefix}{study_id}")
            if isinstance(metrics, dict):
                study_cfg["best_bleu"] = metrics.get("metrics", {}).get("overall_corpus_bleu", "N/A")
                study_cfg["best_meteor"] = metrics.get("metrics", {}).get("mean_meteor", "N/A")

    with open(os.path.join(ROOT_DIR, "master_experiment_ledger.json"), 'w') as f: 
        json.dump(master_ledger, f, indent=4)
    
    generate_csv_file(os.path.join(ROOT_DIR, "word_level_metrics.csv"), master_ledger["word_level"])
    generate_csv_file(os.path.join(ROOT_DIR, "character_level_metrics.csv"), master_ledger["character_level"])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")
    
    eval_p = subparsers.add_parser("evaluate")
    eval_p.add_argument("--checkpoint", type=str, default=None)
    eval_p.add_argument("--test_data", type=str, default=None)
    eval_p.add_argument("--token_type", type=str, default=None)
    eval_p.add_argument("--sample_size", type=float, default=0.2, help="Number of samples (e.g. 500) or fraction (e.g. 0.1 for 10%)")
    eval_p.add_argument("--seed", type=int, default=42, help="Random seed for sampling reproducibility")
    
    viz_p = subparsers.add_parser("visualize")
    viz_p.add_argument("--model", type=str, required=True)
    viz_p.add_argument("--test_data", type=str, default="data/processed/test.csv")
    viz_p.add_argument("--index", type=int, default=0)
    
    compile_p = subparsers.add_parser("compile")
    compile_p.add_argument("--token_type", type=str, default="both")
    
    args = parser.parse_args()
    
    if args.mode == "compile": 
        run_compile_results()
    elif args.mode == "visualize": 
        visualize_sample_attention(args.model, args.test_data, args.index)
    else:
        if getattr(args, 'checkpoint', None): 
            run_evaluation(args.checkpoint, args.test_data, sample_size=getattr(args, 'sample_size', None), seed=getattr(args, 'seed', 42))
        else: 
            evaluate_all_local_models(getattr(args, 'token_type', None), sample_size=getattr(args, 'sample_size', None), seed=getattr(args, 'seed', 42))