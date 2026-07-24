import os
import json
import argparse
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
try:
    from nltk.translate.meteor_score import meteor_score
except ImportError:
    meteor_score = None
from models import Encoder, Decoder, Seq2Seq
from evaluate import translate_sentence

# Enable TensorCore TF32 execution globally for Ampere GPUs
torch.set_float32_matmul_precision('high')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)


class PivotTranslator:
    def __init__(self, de_en_path, en_sv_path, device, token_type="word"):
        self.device = device
        self.token_type = token_type
        print("Loading DE -> EN Model Layout...")
        de_en_checkpoint = torch.load(de_en_path, map_location=device, weights_only=False)
        self.de_en_model, self.de_en_src_vocab, self.de_en_trg_vocab = self._reconstruct_model(de_en_checkpoint)
        del de_en_checkpoint

        print("Loading EN -> SV Model Layout...")
        en_sv_checkpoint = torch.load(en_sv_path, map_location=device, weights_only=False)
        self.en_sv_model, self.en_sv_src_vocab, self.en_sv_trg_vocab = self._reconstruct_model(en_sv_checkpoint)
        del en_sv_checkpoint

    def _reconstruct_model(self, checkpoint):
        config = checkpoint['config']
        src_vocab = checkpoint['src_vocab']
        trg_vocab = checkpoint['trg_vocab']

        is_bidi = config.get('bidirectional', True)
        enc_hidden_dim = config['hidden_dim'] * (2 if is_bidi else 1)

        enc = Encoder(
            len(src_vocab),
            config['emb_dim'],
            config['hidden_dim'],
            config.get('n_layers', 2),
            config['dropout'],
            rnn_type=config['rnn_type'],
            bidirectional=is_bidi,
        )
        dec = Decoder(
            len(trg_vocab),
            config['emb_dim'],
            enc_hidden_dim,
            config['hidden_dim'],
            config.get('n_layers', 2),
            config['dropout'],
            rnn_type=config['rnn_type'],
            attention_type=config.get('attention_type', 'none'),
        )

        model = Seq2Seq(enc, dec, self.device).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Freeze layers and enable optimized cuDNN inference paths
        return model, src_vocab, trg_vocab

    def _tokenize(self, text):
        text = str(text).strip()
        return list(text) if self.token_type == "char" else text.split()

    def _join(self, tokens):
        return "".join(tokens) if self.token_type == "char" else " ".join(tokens)

    def translate(self, de_sentence):
        src_tokens = self._tokenize(de_sentence)

        with torch.no_grad():
            # Stage 1: DE -> EN, using the DE-EN model's own vocab pair.
            en_tokens, _ = translate_sentence(
                self.de_en_model, src_tokens, self.de_en_src_vocab, self.de_en_trg_vocab, self.device
            )
            en_sentence = self._join(en_tokens)

            # Stage 2: EN -> SV. Critically, re-tokenize/numericalize against the EN-SV
            # model's OWN source vocab (en_sv_src_vocab), not the DE-EN model's target
            # vocab (de_en_trg_vocab) - the two were built independently from different
            # training runs, so their word->index mappings don't line up even though
            # both are "English".
            en_tokens_for_sv = self._tokenize(en_sentence)
            sv_tokens, _ = translate_sentence(
                self.en_sv_model, en_tokens_for_sv, self.en_sv_src_vocab, self.en_sv_trg_vocab, self.device
            )
            sv_sentence = self._join(sv_tokens)

        return sv_sentence, en_sentence, en_tokens, sv_tokens


def run_quantitative_evaluation(translator, token_type, experiment, max_samples=None):
    """Runs the DE->EN->SV pivot chain over the aligned pivot evaluation set
    (built by build_pivot_eval_set.py from real Europarl data, not synthetic
    references) and reports corpus BLEU/METEOR for both the final SV output
    and the intermediate EN pivot stage, so the report can show where quality
    is lost across the two-stage chain."""
    import pandas as pd

    eval_csv = os.path.join(ROOT_DIR, "data", "processed", "pivot_de_en_sv_eval.csv")
    if not os.path.exists(eval_csv):
        print(f"\n[ERROR] Pivot evaluation set not found at {eval_csv}.")
        print("Run 'python src/build_pivot_eval_set.py' first to build it from the raw corpora.")
        return

    df = pd.read_csv(eval_csv)
    if max_samples is not None and max_samples < len(df):
        df = df.iloc[:max_samples]

    print(f"\n📊 Running quantitative pivot evaluation on {len(df):,} DE->SV pairs (experiment: {experiment})...")

    smoother = SmoothingFunction().method1
    sv_refs, sv_hyps = [], []
    en_refs, en_hyps = [], []

    for i, row in enumerate(df.itertuples(index=False)):
        de_sentence, en_reference, sv_reference = row.de, row.en, row.sv
        sv_output, en_output, _, _ = translator.translate(de_sentence)

        sv_hyps.append(sv_output.split() if token_type != "char" else list(sv_output))
        sv_refs.append([sv_reference.split() if token_type != "char" else list(sv_reference)])
        en_hyps.append(en_output.split() if token_type != "char" else list(en_output))
        en_refs.append([en_reference.split() if token_type != "char" else list(en_reference)])

        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{len(df)} translated")

    sv_bleu = corpus_bleu(sv_refs, sv_hyps, smoothing_function=smoother) * 100.0
    en_bleu = corpus_bleu(en_refs, en_hyps, smoothing_function=smoother) * 100.0

    sv_meteor = 0.0
    en_meteor = 0.0
    if meteor_score is not None:
        try:
            sv_meteors = [meteor_score([r[0]], h) for r, h in zip(sv_refs, sv_hyps)]
            en_meteors = [meteor_score([r[0]], h) for r, h in zip(en_refs, en_hyps)]
            sv_meteor = (sum(sv_meteors) / len(sv_meteors)) * 100.0 if sv_meteors else 0.0
            en_meteor = (sum(en_meteors) / len(en_meteors)) * 100.0 if en_meteors else 0.0
        except Exception as e:
            print(f"Warning: METEOR computation failed: {e}")

    print(f"\n✅ Pivot evaluation complete ({len(df):,} sentences):")
    print(f"  DE -> EN (intermediate stage): BLEU={en_bleu:.2f} METEOR={en_meteor:.2f}")
    print(f"  DE -> EN -> SV (final output): BLEU={sv_bleu:.2f} METEOR={sv_meteor:.2f}")

    results = {
        "experiment": experiment,
        "token_type": token_type,
        "attention_type": "pivot",
        "rnn_type": "PIVOT",
        "embedding_source": "n/a",
        "n_samples": len(df),
        "intermediate_en_bleu": round(en_bleu, 2),
        "intermediate_en_meteor": round(en_meteor, 2),
        "bleu": round(sv_bleu, 2),
        "meteor": round(sv_meteor, 2),
        "eval_split": "pivot_aligned_eval_set",
    }

    output_dir = os.path.join(ROOT_DIR, "data", "results")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"best_config_{experiment}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Shot German to Swedish via English Pivot")
    parser.add_argument("--de_en_model", type=str, required=True)
    parser.add_argument("--en_sv_model", type=str, required=True)
    parser.add_argument("--text", type=str, default=None, help="Text sentence to translate")
    parser.add_argument("--evaluate", action="store_true", help="Run quantitative evaluation mode")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap the number of pivot evaluation pairs")
    parser.add_argument("--token_type", type=str, default="word", choices=["word", "char"])
    parser.add_argument("--experiment", type=str, default="PIVOT")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    translator = PivotTranslator(args.de_en_model, args.en_sv_model, device, token_type=args.token_type)

    if args.text:
        sv_output, intermediate_en, _, _ = translator.translate(args.text)
        print(f"\nOrigin (DE):      {args.text}")
        print(f"Pivot (EN):       {intermediate_en}")
        print(f"Output (SV):      {sv_output}")

    if args.evaluate:
        run_quantitative_evaluation(translator, args.token_type, args.experiment, max_samples=args.max_samples)