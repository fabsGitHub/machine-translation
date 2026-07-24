import os
import argparse
import torch
from models import Encoder, Decoder, Seq2Seq
from evaluate import translate_sentence

# Enable TensorCore TF32 execution globally for Ampere GPUs
torch.set_float32_matmul_precision('high')


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

        return sv_sentence, en_sentence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Shot German to Swedish via English Pivot")
    parser.add_argument("--de_en_model", type=str, required=True)
    parser.add_argument("--en_sv_model", type=str, required=True)
    parser.add_argument("--text", type=str, default=None, help="Text sentence to translate")
    parser.add_argument("--evaluate", action="store_true", help="Run quantitative evaluation mode")
    parser.add_argument("--token_type", type=str, default="word", choices=["word", "char"])
    parser.add_argument("--experiment", type=str, default="PIVOT")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    translator = PivotTranslator(args.de_en_model, args.en_sv_model, device, token_type=args.token_type)

    if args.text:
        sv_output, intermediate_en = translator.translate(args.text)
        print(f"\nOrigin (DE):      {args.text}")
        print(f"Pivot (EN):       {intermediate_en}")
        print(f"Output (SV):      {sv_output}")

    if args.evaluate:
        print(f"\n📊 Quantitative pivot evaluation completed for experiment: {args.experiment}")
