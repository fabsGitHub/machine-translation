import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
try:
    from nltk.translate.meteor_score import meteor_score
except ImportError:
    meteor_score = None

# Internal Module Imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)

from dataset import get_dataloader, PAD_IDX, SOS_IDX, EOS_IDX
from models import Encoder, Decoder, Seq2Seq
from config import load_config

# Ensure required NLTK resources are available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


# ------------------------------------------------------------------------
# Helper Utilities & Inference
# ------------------------------------------------------------------------

def idx_to_tokens(indices, vocab):
    """Converts token indices back to readable string tokens."""
    if hasattr(vocab, 'get_itos'):
        itos = vocab.get_itos()
        tokens = [itos[i] for i in indices]
    elif hasattr(vocab, 'itos'):
        tokens = [vocab.itos[i] for i in indices]
    elif isinstance(vocab, dict):
        inv_vocab = {v: k for k, v in vocab.items()}
        tokens = [inv_vocab.get(i, "<unk>") for i in indices]
    else:
        tokens = [str(i) for i in indices]
    return tokens


def build_model_from_checkpoint(checkpoint, device):
    """Reconstructs the Seq2Seq model architecture from checkpoint metadata."""
    cfg = checkpoint['config']
    src_vocab = checkpoint['src_vocab']
    trg_vocab = checkpoint['trg_vocab']

    num_directions = 2 if cfg.get('bidirectional', True) else 1
    emb_dim = cfg.get('emb_dim', 256)
    hidden_dim = cfg.get('hidden_dim', 512)
    dropout = cfg.get('dropout', 0.3)
    rnn_type = cfg.get('rnn_type', 'LSTM')
    attention_type = cfg.get('attention_type', 'none')
    embedding_source = cfg.get('embedding_source', 'scratch')
    freeze_emb = cfg.get('freeze_emb', False)

    emb_override = 300 if embedding_source == 'glove' else None

    encoder = Encoder(
        len(src_vocab), emb_dim, hidden_dim, 2, dropout,
        rnn_type, cfg.get('bidirectional', True), None, freeze_emb, emb_override
    )
    decoder = Decoder(
        len(trg_vocab), emb_dim, hidden_dim * num_directions, hidden_dim, 2,
        dropout, rnn_type, attention_type, None, freeze_emb, emb_override
    )

    model = Seq2Seq(encoder, decoder, device).to(device)

    clean_state_dict = {
        k.replace("_orig_mod.", "").replace("module.", ""): v
        for k, v in checkpoint['model_state_dict'].items()
    }
    model.load_state_dict(clean_state_dict)
    model.eval()
    return model, src_vocab, trg_vocab, cfg


def translate_sentence(model, src_tokens, src_vocab, trg_vocab, device, max_len=50):
    """Translates a source sequence and captures target output and attention matrix."""
    model.eval()

    # Numericalize source
    if hasattr(src_vocab, '__getitem__'):
        src_indices = [SOS_IDX] + [src_vocab[tok] if tok in src_vocab else src_vocab.get('<unk>', 0) for tok in src_tokens] + [EOS_IDX]
    else:
        src_indices = [SOS_IDX] + [src_vocab.get(tok, 0) for tok in src_tokens] + [EOS_IDX]

    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)

    with torch.no_grad():
        if hasattr(model.encoder, 'get_encoder'):
            encoder_outputs, hidden = model.encoder(src_tensor)
        else:
            encoder_outputs, hidden = model.encoder(src_tensor)

        trg_indexes = [SOS_IDX]
        attentions = []

        for _ in range(max_len):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
            output, hidden, attn = model.decoder(trg_tensor, hidden, encoder_outputs)

            if attn is not None:
                attentions.append(attn.squeeze(0).cpu().detach().numpy())

            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)

            if pred_token == EOS_IDX:
                break

    translated_tokens = idx_to_tokens(trg_indexes[1:], trg_vocab)
    if translated_tokens and translated_tokens[-1] == "<eos>":
        translated_tokens = translated_tokens[:-1]

    attn_matrix = np.array(attentions) if len(attentions) > 0 else None
    return translated_tokens, attn_matrix


# ------------------------------------------------------------------------
# Primary Required Functions
# ------------------------------------------------------------------------

def visualize_attention(model_path, src_sentence=None, save_path=None, device=None):
    """Generates and saves an attention heatmap for a sample input sentence."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        print(f"❌ Model checkpoint missing: {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model, src_vocab, trg_vocab, cfg = build_model_from_checkpoint(checkpoint, device)

    if cfg.get("attention_type", "none") == "none":
        print(f"⚠️ Model at {model_path} does not use attention (attention_type='none'). Skipping visualization.")
        return

    if src_sentence is None:
        src_sentence = "ein kleiner hund läuft über den rasen ."

    token_type = cfg.get("token_type", "word")
    src_tokens = list(src_sentence) if token_type == "char" else src_sentence.strip().split()

    translated_tokens, attn_matrix = translate_sentence(model, src_tokens, src_vocab, trg_vocab, device)

    if attn_matrix is None or attn_matrix.size == 0:
        print("⚠️ No attention weights captured during inference.")
        return

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn_matrix[:len(translated_tokens), :len(src_tokens) + 2],
        xticklabels=["<sos>"] + src_tokens + ["<eos>"],
        yticklabels=translated_tokens,
        cmap="viridis",
        annot=False
    )
    plt.xlabel("Source Sequence")
    plt.ylabel("Target Sequence")
    plt.title(f"Attention Map ({cfg.get('experiment', 'NMT')} - {cfg.get('attention_type', 'Luong').upper()})")
    plt.tight_layout()

    if save_path is None:
        exp_name = cfg.get('experiment', 'attention_map')
        save_path = os.path.join(ROOT_DIR, "data", "results", f"{exp_name}_attention.png")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Attention heatmap visualization saved to: {save_path}")


def generate_all_reports(token_type="word", output_dir=None):
    """Aggregates all experiment JSON outputs into unified summary tables (CSV and JSON)."""
    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "data", "results")

    json_files = glob.glob(os.path.join(output_dir, "best_config_*.json"))
    if not json_files:
        print(f"⚠️ No result json logs found in {output_dir}")
        return None

    records = []
    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            if token_type and data.get("token_type", "word") != token_type:
                continue

            records.append({
                "Experiment": data.get("experiment", "N/A"),
                "RNN Type": data.get("rnn_type", "LSTM"),
                "Attention": data.get("attention_type", "none"),
                "Token Type": data.get("token_type", "word"),
                "Embedding": data.get("embedding_source", "scratch"),
                "BLEU": data.get("bleu", data.get("bleu_score", None)),
                "METEOR": data.get("meteor", data.get("mean_meteor", None)),
                "Best Val Loss": data.get("best_val_loss", None),
                "Epochs Trained": data.get("epochs_trained", None),
                "Train Time": data.get("train_time", "N/A"),
                "Inference Time": data.get("inference_time", "N/A")
            })
        except Exception as e:
            print(f"⚠️ Error loading {filepath}: {e}")

    if not records:
        print(f"⚠️ No records matched token_type='{token_type}'.")
        return None

    df = pd.DataFrame(records)
    if "BLEU" in df.columns and df["BLEU"].notnull().any():
        df = df.sort_values(by="BLEU", ascending=False)

    summary_csv = os.path.join(output_dir, f"evaluation_report_{token_type}.csv")
    summary_json = os.path.join(output_dir, f"evaluation_report_{token_type}.json")

    df.to_csv(summary_csv, index=False)
    df.to_json(summary_json, orient="records", indent=4)

    print("\n" + "=" * 80)
    print(f"📊 SUMMARY EVALUATION REPORT ({token_type.upper()} LEVEL)")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    print(f"📁 Summary report written to: {summary_csv}\n")

    return df


# ------------------------------------------------------------------------
# Evaluation Pipeline
# ------------------------------------------------------------------------

def evaluate_checkpoint(checkpoint_path, max_samples=1000, device=None):
    """Evaluates BLEU and METEOR metrics for a saved checkpoint on test/val set."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, src_vocab, trg_vocab, cfg = build_model_from_checkpoint(checkpoint, device)

    processed_dir = os.path.join(ROOT_DIR, "data", "processed")
    src_lang = cfg.get("src_lang", "de")
    trg_lang = cfg.get("trg_lang", "en")
    token_type = cfg.get("token_type", "word")

    test_csv = os.path.join(processed_dir, f"val_{src_lang}_{trg_lang}.csv")
    if not os.path.exists(test_csv):
        test_csv = os.path.join(processed_dir, "val.csv")

    test_loader, _, _ = get_dataloader(
        test_csv, batch_size=32, shuffle=False,
        src_vocab=src_vocab, trg_vocab=trg_vocab,
        src_lang=src_lang, trg_lang=trg_lang, token_type=token_type
    )

    targets = []
    hypotheses = []
    meteor_scores = []

    count = 0
    smoother = SmoothingFunction().method1

    for src_batch, trg_batch in test_loader:
        if count >= max_samples:
            break

        for i in range(src_batch.size(0)):
            if count >= max_samples:
                break

            src_idxs = [idx.item() for idx in src_batch[i] if idx.item() not in (PAD_IDX, SOS_IDX, EOS_IDX)]
            trg_idxs = [idx.item() for idx in trg_batch[i] if idx.item() not in (PAD_IDX, SOS_IDX, EOS_IDX)]

            src_tokens = idx_to_tokens(src_idxs, src_vocab)
            trg_tokens = idx_to_tokens(trg_idxs, trg_vocab)

            pred_tokens, _ = translate_sentence(model, src_tokens, src_vocab, trg_vocab, device)

            hypotheses.append(pred_tokens)
            targets.append([trg_tokens])

            if meteor_score is not None:
                try:
                    ref_str = " ".join(trg_tokens)
                    hyp_str = " ".join(pred_tokens)
                    meteor_scores.append(meteor_score([ref_str.split()], hyp_str.split()))
                except Exception:
                    pass

            count += 1

    bleu = corpus_bleu(targets, hypotheses, smoothing_function=smoother) * 100.0
    mean_meteor = (sum(meteor_scores) / len(meteor_scores) * 100.0) if meteor_scores else 0.0

    print(f"BLEU: {bleu:.4f}")
    print(f"METEOR: {mean_meteor:.4f}")

    return bleu, mean_meteor


# ------------------------------------------------------------------------
# CLI Entry Point
# ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluation and Reporting Interface")
    parser.add_argument("mode", choices=["evaluate", "report", "visualize"], nargs="?", default="report")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max test samples for BLEU evaluation")
    parser.add_argument("--token_type", type=str, default="word", choices=["word", "char"])
    parser.add_argument("--sentence", type=str, default=None, help="Sample sentence for attention visualization")
    args = parser.parse_args()

    if args.mode == "evaluate":
        if not args.checkpoint:
            print("❌ --checkpoint is required for 'evaluate' mode.")
            sys.exit(1)
        evaluate_checkpoint(args.checkpoint, max_samples=args.max_samples)

    elif args.mode == "visualize":
        if not args.checkpoint:
            print("❌ --checkpoint is required for 'visualize' mode.")
            sys.exit(1)
        visualize_attention(args.checkpoint, src_sentence=args.sentence)

    elif args.mode == "report":
        generate_all_reports(token_type=args.token_type)


if __name__ == "__main__":
    main()