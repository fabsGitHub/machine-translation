import random
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import pad_vocab_size

# ============================================================================
# JIT-COMPILED CUDA KERNELS
# ============================================================================

@torch.jit.script
def lstm_decoder_step(input_token: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor,
                      weight_ih: torch.Tensor, weight_hh: torch.Tensor,
                      bias_ih: torch.Tensor, bias_hh: torch.Tensor):
    """
    Fused JIT CUDA kernel for individual recurrent decoder step execution.
    Eliminates Python runtime overhead between steps.
    """
    return torch.lstm_cell(input_token, (hidden, cell), weight_ih, weight_hh, bias_ih, bias_hh)

# ============================================================================
# ATTENTION STRATEGIES
# ============================================================================

class AttentionStrategy(nn.Module, ABC):
    @abstractmethod
    def forward(self, decoder_hidden, encoder_outputs, projected_encoder=None, mask=None):
        pass

class NoAttention(AttentionStrategy):
    def forward(self, decoder_hidden, encoder_outputs, projected_encoder=None, mask=None):
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        tgt_len = decoder_hidden.size(1) if decoder_hidden.dim() == 3 else 1
        return torch.zeros(batch_size, tgt_len, src_len, device=encoder_outputs.device)

class LuongAttention(AttentionStrategy):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.attn = nn.Linear(encoder_hidden_dim, decoder_hidden_dim)
        
    def forward(self, decoder_hidden, encoder_outputs, projected_encoder=None, mask=None):
        if projected_encoder is None:
            projected_encoder = self.attn(encoder_outputs)
            
        if decoder_hidden.dim() == 2:
            decoder_hidden = decoder_hidden.unsqueeze(1)
            
        scores = torch.bmm(decoder_hidden, projected_encoder.permute(0, 2, 1))
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e4)
        return torch.softmax(scores, dim=-1)

class BahdanauAttention(AttentionStrategy):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.W_a = nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
        self.U_a = nn.Linear(encoder_hidden_dim, decoder_hidden_dim)
        self.v_a = nn.Linear(decoder_hidden_dim, 1, bias=False)
        
    def forward(self, decoder_hidden, encoder_outputs, projected_encoder=None, mask=None):
        keys = projected_encoder if projected_encoder is not None else self.U_a(encoder_outputs)
        if decoder_hidden.dim() == 2:
            query = self.W_a(decoder_hidden).unsqueeze(1)
            energy = torch.tanh(query.unsqueeze(2) + keys.unsqueeze(1))
            scores = self.v_a(energy).squeeze(-1)
        else:
            query = self.W_a(decoder_hidden)
            energy = torch.tanh(query.unsqueeze(2) + keys.unsqueeze(1))
            scores = self.v_a(energy).squeeze(-1)
            
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e4)
        return torch.softmax(scores, dim=-1)

# ============================================================================
# ENCODER MODULE
# ============================================================================

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout, 
                 rnn_type="LSTM", bidirectional=True, pretrained_embeddings=None, 
                 freeze_embeddings=False, pretrained_dim=None):
        super().__init__()
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        
        # Vocab padded to multiple of 16 for Tensor Core alignment
        padded_input_dim = pad_vocab_size(input_dim, multiple=16)
        actual_emb_dim = emb_dim

        if pretrained_embeddings is not None:
            # Zero-pad pretrained embedding rows to match padded_input_dim
            if pretrained_embeddings.shape[0] < padded_input_dim:
                pad_rows = padded_input_dim - pretrained_embeddings.shape[0]
                padding_tensor = torch.zeros(
                    (pad_rows, pretrained_embeddings.shape[1]), 
                    dtype=pretrained_embeddings.dtype, 
                    device=pretrained_embeddings.device
                )
                pretrained_embeddings = torch.cat([pretrained_embeddings, padding_tensor], dim=0)
            
            actual_emb_dim = pretrained_embeddings.shape[1]
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=freeze_embeddings, padding_idx=0
            )
        elif pretrained_dim is not None:
            actual_emb_dim = pretrained_dim
            self.embedding = nn.Embedding(padded_input_dim, actual_emb_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(padded_input_dim, emb_dim, padding_idx=0)
            
        if actual_emb_dim != emb_dim:
            self.project = nn.Linear(actual_emb_dim, emb_dim)
        else:
            self.project = nn.Identity()
            
        rnn_cls = getattr(nn, rnn_type)
        self.rnn = rnn_cls(
            emb_dim, hidden_dim, num_layers=n_layers, 
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.project(self.embedding(src)))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden


# ============================================================================
# DECODER MODULE
# ============================================================================

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, encoder_hidden_dim, decoder_hidden_dim, 
                 n_layers, dropout, rnn_type="LSTM", attention_type="none", 
                 pretrained_embeddings=None, freeze_embeddings=False, pretrained_dim=None):
        super().__init__()
        self.output_dim = output_dim
        self.padded_output_dim = pad_vocab_size(output_dim, multiple=16)
        self.rnn_type = rnn_type
        self.attention_type = attention_type
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        
        if pretrained_embeddings is not None:
            # Zero-pad pretrained embedding rows to match padded_output_dim
            if pretrained_embeddings.shape[0] < self.padded_output_dim:
                pad_rows = self.padded_output_dim - pretrained_embeddings.shape[0]
                padding_tensor = torch.zeros(
                    (pad_rows, pretrained_embeddings.shape[1]), 
                    dtype=pretrained_embeddings.dtype, 
                    device=pretrained_embeddings.device
                )
                pretrained_embeddings = torch.cat([pretrained_embeddings, padding_tensor], dim=0)

            actual_emb_dim = pretrained_embeddings.shape[1]
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=freeze_embeddings, padding_idx=0
            )
        else:
            actual_emb_dim = pretrained_dim if pretrained_dim is not None else emb_dim
            self.embedding = nn.Embedding(self.padded_output_dim, actual_emb_dim, padding_idx=0)

        if actual_emb_dim != emb_dim:
            self.project = nn.Linear(actual_emb_dim, emb_dim)
        else:
            self.project = nn.Identity()

        if attention_type == "luong":
            self.attention = LuongAttention(encoder_hidden_dim, decoder_hidden_dim)
        elif attention_type == "bahdanau":
            self.attention = BahdanauAttention(encoder_hidden_dim, decoder_hidden_dim)
        else:
            self.attention = NoAttention()

        rnn_in_dim = emb_dim + (encoder_hidden_dim if attention_type != "none" else 0)
        rnn_cls = getattr(nn, rnn_type)
        self.rnn = rnn_cls(
            rnn_in_dim, decoder_hidden_dim, num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0, batch_first=True
        )
        
        self.fc_out = nn.Linear(decoder_hidden_dim, self.padded_output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_step, hidden, encoder_outputs):
        """
        Executes a single autoregressive decoding step.
        """
        if input_step.dim() == 1:
            input_step = input_step.unsqueeze(1)  # [batch_size, 1]

        embedded = self.dropout(self.project(self.embedding(input_step)))  # [batch_size, 1, emb_dim]

        # Extract top-layer hidden state query across LSTM (h, c) or GRU/RNN h
        decoder_query = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]

        if self.attention_type != "none":
            attn_weights = self.attention(decoder_query, encoder_outputs)  # [batch_size, 1, src_len]
            context = torch.bmm(attn_weights, encoder_outputs)              # [batch_size, 1, encoder_hidden_dim]
            rnn_input = torch.cat((embedded, context), dim=2)               # [batch_size, 1, emb_dim + encoder_hidden_dim]
        else:
            rnn_input = embedded

        rnn_output, hidden = self.rnn(rnn_input, hidden)  # rnn_output: [batch_size, 1, decoder_hidden_dim]
        output = self.fc_out(rnn_output.squeeze(1))       # output: [batch_size, padded_output_dim]
        
        return output, hidden

    def forward_vectorized(self, trg_input, hidden, encoder_outputs):
        """
        Executes parallelized full-sequence forward pass when teacher_forcing_ratio == 1.0.
        """
        embedded = self.dropout(self.project(self.embedding(trg_input)))

        if self.attention_type != "none":
            # For attention models, process step-by-step across the sequence length
            outputs = []
            current_hidden = hidden
            for t in range(trg_input.size(1)):
                step_in = trg_input[:, t]
                out, current_hidden = self.forward_step(step_in, current_hidden, encoder_outputs)
                outputs.append(out.unsqueeze(1))
            return torch.cat(outputs, dim=1), current_hidden
        else:
            # Non-attention models can process the entire sequence in a single parallel RNN pass
            rnn_output, hidden = self.rnn(embedded, hidden)
            predictions = self.fc_out(rnn_output)
            return predictions, hidden
             
# ============================================================================
# SEQ2SEQ WRAPPER
# ============================================================================

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def _bridge_hidden(self, hidden):
        if self.encoder.bidirectional:
            if self.encoder.rnn_type == "LSTM":
                h, c = hidden
                h = h.view(h.size(0) // 2, 2, h.size(1), h.size(2)).sum(dim=1)
                c = c.view(c.size(0) // 2, 2, c.size(1), c.size(2)).sum(dim=1)
                return (h, c)
            else:
                return hidden.view(hidden.size(0) // 2, 2, hidden.size(1), hidden.size(2)).sum(dim=1)
        return hidden

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        
        encoder_outputs, hidden = self.encoder(src)
        hidden = self._bridge_hidden(hidden)

        # ⚡ OPTIMIZATION: Zero-overhead padding for Fully Vectorized Matrix Pass during Teacher Forcing
        if teacher_forcing_ratio == 1.0:
            trg_input = trg[:, :-1]
            predictions, _ = self.decoder.forward_vectorized(trg_input, hidden, encoder_outputs)
            # Efficiently pad time dimension on the left without creating & slicing extra tensors
            return F.pad(predictions, (0, 0, 1, 0))

        # Sequential decoding fallback for partial teacher forcing / inference
        outputs = torch.zeros(batch_size, trg_len, self.decoder.padded_output_dim, device=self.device)
        input_step = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden = self.decoder.forward_step(input_step, hidden, encoder_outputs)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_step = trg[:, t] if teacher_force else top1

        return outputs