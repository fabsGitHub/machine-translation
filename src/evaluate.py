import os
import sys
import json
import re
import csv
import time
import glob
import multiprocessing
import argparse
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


def _worker_meteor_chunk(pairs_chunk):
    """Processes a chunk of reference/hypothesis pairs to eliminate granular IPC pickling overhead."""
    return [meteor_score(ref, hyp) for ref, hyp in pairs_chunk]


def translate_sentence(model, src_tensor, trg_vocab, device, max_len=50):
    model.eval()
    tokens = []
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        hidden = model._bridge_hidden(hidden)
        
        current_token = torch.tensor([SOS_IDX], dtype=torch.long, device=device)
        
        for _ in range(max_len):
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
    print(f"⌛ Loading checkpoint for evaluation: '{os.path.basename(checkpoint_path)}'")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    state_dict = checkpoint['model_state_dict']
    src_vocab = checkpoint.get('src_vocab')
    trg_vocab = checkpoint.get('trg_vocab')

    src_lang, trg_lang = config.get('src_lang', 'en'), config.get('trg_lang', 'de')
    token_type, experiment_id = config.get('token_type', 'word'), config.get('experiment', 'Unknown')

    max_len = 250 if token_type == "char" else 50

    if not test_csv:
        test_csv = os.path.join(ROOT_DIR, "data", "processed", f"test_{src_lang}_{trg_lang}.csv")
        if not os.path.exists(test_csv):
            legacy_file = "test_sv.csv" if ("sv" in (src_lang, trg_lang)) else "test.csv"
            test_csv = os.path.join(ROOT_DIR, "data", "processed", legacy_file)
            
    if not os.path.exists(test_csv): 
        return
        
    test_loader, _, _ = get_dataloader(
        test_csv, batch_size=config.get('batch_size', 2048), shuffle=False, 
        src_vocab=src_vocab, trg_vocab=trg_vocab, src_lang=src_lang, trg_lang=trg_lang, token_type=token_type
    )
    
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
    
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        model = torch.compile(model)
    else:
        print("Skipping torch.compile: GPU Compute Capability < 7.0")
    
    references, hypotheses = [], []
    meta_info = []
    
    start_time = time.time()
    for src, trg in test_loader:
        src = src.to(device, non_blocking=True)
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
    
    print("📊 Computing METEOR scores in batched chunks across CPU cores...")
    meteor_pairs = list(zip(references, hypotheses))
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # OPTIMIZATION: Chunk dataset list into larger batches to minimize IPC serialization overhead
    chunk_size = max(100, len(meteor_pairs) // (num_workers * 4))
    chunks = [meteor_pairs[i:i + chunk_size] for i in range(0, len(meteor_pairs), chunk_size)]
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        chunk_results = pool.map(_worker_meteor_chunk, chunks)
        
    meteor_scores = [score for sublist in chunk_results for score in sublist]
        
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
            with open(ledger_path, 'r', encoding='utf-8') as f:
                ledger_data = json.load(f)
        except Exception:
            ledger_data = {}

    ledger_data[experiment_id] = {
        "experiment": experiment_id,
        "token_type": token_type,
        "rnn_type": config.get("rnn_type"),
        "bidirectional": is_bidi,
        "attention_type": config.get("attention_type", "none"),
        "embedding_source": config.get("embedding_source", "scratch"),
        "metrics": {
            "overall_corpus_bleu": round(bleu_score, 2),
            "mean_meteor": round(mean_meteor, 4),
            "bucket_analysis": bucket_analysis_results
        },
        "bleu": round(bleu_score, 2),
        "meteor": round(mean_meteor, 4),
        "avg_inference_ms": round(avg_inference_ms, 2)
    }

    with open(ledger_path, 'w', encoding='utf-8') as f:
        json.dump(ledger_data, f, indent=4)

    print(f"✅ Evaluation complete. Metrics saved to {ledger_path}")


def compile_reports(token_type="word"):
    from train_pipeline import generate_all_reports
    generate_all_reports(token_type)


def visualize_attention(model_path, sample_text=None):
    print(f"📊 Attention visualization routine initialized for: {os.path.basename(model_path)}")


def main():
    parser = argparse.ArgumentParser(description="NMT Evaluation & Analysis Interface")
    subparsers = parser.add_subparsers(dest="mode")

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--checkpoint", type=str, required=False)
    eval_parser.add_argument("--token_type", type=str, default="word")
    eval_parser.add_argument("--sample_size", type=float, default=None)

    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--token_type", type=str, default="word")

    vis_parser = subparsers.add_parser("visualize")
    vis_parser.add_argument("--model", type=str, required=True)
    vis_parser.add_argument("--text", type=str, default=None)

    args = parser.parse_args()

    if args.mode == "evaluate":
        if args.checkpoint:
            run_evaluation(args.checkpoint, sample_size=args.sample_size)
        else:
            for pt_file in glob.glob(os.path.join(OUTPUT_DIR, "*.pt")):
                run_evaluation(pt_file, sample_size=args.sample_size)
    elif args.mode == "compile":
        compile_reports(args.token_type)
    elif args.mode == "visualize":
        visualize_attention(args.model, args.text)


if __name__ == "__main__":
    main()