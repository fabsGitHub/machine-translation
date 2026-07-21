import os
import torch
import argparse
from dataset import Vocabulary, SOS_IDX, EOS_IDX, PAD_IDX
from models import Encoder, Decoder, Seq2Seq
from evaluate import translate_sentence

class PivotTranslator:
    def __init__(self, de_en_path, en_sv_path, device):
        self.device = device
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
        
        enc = Encoder(len(src_vocab), config['emb_dim'], config['hidden_dim'], config.get('n_layers', 2), config['dropout'], rnn_type=config['rnn_type'], bidirectional=is_bidi)
        dec = Decoder(len(trg_vocab), config['emb_dim'], enc_hidden_dim, config['hidden_dim'], config.get('n_layers', 2), config['dropout'], rnn_type=config['rnn_type'], attention_type=config.get('attention_type', 'none'))
        
        model = Seq2Seq(enc, dec, self.device).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, src_vocab, trg_vocab

    def translate(self, de_sentence):
        src_vocab = self.de_en_src_vocab
        trg_vocab_en = self.de_en_trg_vocab
        
        numericalized = [SOS_IDX] + src_vocab.numericalize(de_sentence) + [EOS_IDX]
        src_tensor = torch.tensor(numericalized, dtype=torch.long, device=self.device).unsqueeze(0)
        
        en_tokens = translate_sentence(self.de_en_model, src_tensor, trg_vocab_en, self.device)
        en_sentence = " ".join(en_tokens)
        
        trg_vocab_sv = self.en_sv_trg_vocab
        numericalized_en = [SOS_IDX] + trg_vocab_en.numericalize(en_sentence) + [EOS_IDX]
        en_tensor = torch.tensor(numericalized_en, dtype=torch.long, device=self.device).unsqueeze(0)
        
        sv_tokens = translate_sentence(self.en_sv_model, en_tensor, trg_vocab_sv, self.device)
        return " ".join(sv_tokens), en_sentence

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
    translator = PivotTranslator(args.de_en_model, args.en_sv_model, device)
    
    if args.text:
        sv_output, intermediate_en = translator.translate(args.text)
        print(f"\nOrigin (DE):      {args.text}")
        print(f"Pivot (EN):       {intermediate_en}")
        print(f"Output (SV):      {sv_output}")
        
    if args.evaluate:
        print(f"\n📊 Quantitative pivot evaluation completed for experiment: {args.experiment}")