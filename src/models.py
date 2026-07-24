import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# Enable TensorCore TF32 execution globally for Ampere GPUs
torch.set_float32_matmul_precision('high')


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        hidden_dim,
        n_layers=2,
        dropout=0.3,
        rnn_type="LSTM",
        bidirectional=True,
        pretrained_emb=None,
        freeze_emb=False,
        custom_emb_dim=None,
    ):
        super().__init__()
        self.rnn_type = rnn_type
        emb_dim_in = custom_emb_dim if custom_emb_dim else emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim_in)

        if pretrained_emb is not None:
            pretrained_tensor = (
                pretrained_emb
                if isinstance(pretrained_emb, torch.Tensor)
                else torch.as_tensor(pretrained_emb, dtype=torch.float32)
            )
            self.embedding.weight.data[: pretrained_tensor.size(0)].copy_(pretrained_tensor)
            if freeze_emb:
                self.embedding.weight.requires_grad = False

        self.project = (
            nn.Linear(emb_dim_in, emb_dim)
            if custom_emb_dim and custom_emb_dim != emb_dim
            else None
        )
        self.dropout = nn.Dropout(dropout)

        rnn_cls = getattr(nn, rnn_type)
        self.rnn = rnn_cls(
            emb_dim,
            hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        if self.project is not None:
            embedded = self.project(embedded)

        outputs, hidden = self.rnn(embedded)
        return outputs, hidden


class LuongAttention(nn.Module):
    def __init__(self, hidden_dim, enc_hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, enc_hidden_dim)

    def forward(self, hidden, encoder_outputs):
        score = torch.bmm(
            encoder_outputs, self.attn(hidden).unsqueeze(2)
        ).squeeze(2)
        return F.softmax(score, dim=1)


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim, enc_hidden_dim):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, hidden_dim)
        self.U_a = nn.Linear(enc_hidden_dim, hidden_dim)
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, proj_enc_outputs=None):
        if proj_enc_outputs is None:
            proj_enc_outputs = self.U_a(encoder_outputs)

        energy = torch.tanh(self.W_a(hidden).unsqueeze(1) + proj_enc_outputs)
        score = self.v_a(energy).squeeze(2)
        return F.softmax(score, dim=1)


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        enc_hidden_dim,
        hidden_dim,
        n_layers=2,
        dropout=0.3,
        rnn_type="LSTM",
        attention_type="none",
        pretrained_emb=None,
        freeze_emb=False,
        custom_emb_dim=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.attention_type = attention_type
        self.rnn_type = rnn_type
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        emb_dim_in = custom_emb_dim if custom_emb_dim else emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim_in)

        if pretrained_emb is not None:
            pretrained_tensor = (
                pretrained_emb
                if isinstance(pretrained_emb, torch.Tensor)
                else torch.as_tensor(pretrained_emb, dtype=torch.float32)
            )
            self.embedding.weight.data[: pretrained_tensor.size(0)].copy_(pretrained_tensor)
            if freeze_emb:
                self.embedding.weight.requires_grad = False

        self.project = (
            nn.Linear(emb_dim_in, emb_dim)
            if custom_emb_dim and custom_emb_dim != emb_dim
            else None
        )
        self.dropout = nn.Dropout(dropout)

        if attention_type == "luong":
            self.attention = LuongAttention(hidden_dim, enc_hidden_dim)
        elif attention_type == "bahdanau":
            self.attention = BahdanauAttention(hidden_dim, enc_hidden_dim)
        else:
            self.attention = None

        rnn_in_dim = emb_dim + (
            enc_hidden_dim if attention_type != "none" else 0
        )
        rnn_cls = getattr(nn, rnn_type)
        self.rnn = rnn_cls(
            rnn_in_dim,
            hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )

        fc_in_dim = hidden_dim + (
            enc_hidden_dim if attention_type != "none" else 0
        )
        self.fc_out = nn.Linear(fc_in_dim, vocab_size)

    def forward_step(self, input_token, hidden, encoder_outputs, proj_enc_outputs=None):
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))
        if self.project is not None:
            embedded = self.project(embedded)

        if self.attention_type != "none":
            h_top = (
                hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]
            )
            if self.attention_type == "bahdanau":
                attn_weights = self.attention(h_top, encoder_outputs, proj_enc_outputs=proj_enc_outputs)
            else:
                attn_weights = self.attention(h_top, encoder_outputs)

            context = torch.bmm(
                attn_weights.unsqueeze(1), encoder_outputs
            )
            rnn_in = torch.cat((embedded, context), dim=2)
        else:
            context = None
            attn_weights = None
            rnn_in = embedded

        output, hidden = self.rnn(rnn_in, hidden)

        if self.attention_type != "none":
            fc_in = torch.cat((output, context), dim=2)
        else:
            fc_in = output

        prediction = self.fc_out(fc_in.squeeze(1))
        return prediction, hidden, attn_weights

    def forward(
        self, trg, hidden, encoder_outputs, teacher_forcing_ratio=0.4
    ):
        batch_size, trg_len = trg.shape

        proj_enc = (
            self.attention.U_a(encoder_outputs)
            if self.attention_type == "bahdanau"
            else None
        )

        # Fully traceable teacher-forcing check for non-attention mode
        use_teacher_forcing = self.training and (
            (torch.rand(1, device=trg.device).item() < teacher_forcing_ratio)
            if teacher_forcing_ratio > 0.0
            else False
        )

        if use_teacher_forcing and self.attention_type == "none":
            trg_in = trg[:, :-1]
            embedded = self.dropout(self.embedding(trg_in))
            if self.project is not None:
                embedded = self.project(embedded)

            rnn_out, _ = self.rnn(embedded, hidden)
            predictions = self.fc_out(rnn_out)

            zero_step = torch.zeros(
                (batch_size, 1, self.vocab_size), device=trg.device, dtype=predictions.dtype
            )
            return torch.cat([zero_step, predictions], dim=1)

        # Pre-allocate output tensor directly in VRAM
        outputs = torch.zeros(
            batch_size, trg_len, self.vocab_size, device=trg.device, dtype=encoder_outputs.dtype
        )
        input_token = trg[:, 0]

        # Generate teacher-forcing mask as a GPU boolean Tensor (1D)
        if self.training and teacher_forcing_ratio > 0.0:
            use_tf_steps = torch.rand(trg_len - 1, device=trg.device) < teacher_forcing_ratio
        else:
            use_tf_steps = torch.zeros(trg_len - 1, dtype=torch.bool, device=trg.device)

        for t in range(1, trg_len):
            pred, hidden, _ = self.forward_step(
                input_token, hidden, encoder_outputs, proj_enc_outputs=proj_enc
            )
            outputs[:, t] = pred
            next_tf = trg[:, t]

            # Vectorized selection via torch.where (Traceable by torch.compile)
            input_token = torch.where(use_tf_steps[t - 1], next_tf, pred.argmax(dim=1))

        return outputs

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # Device attribute removed; retrieve dynamically from parameters when needed

        if encoder.rnn.bidirectional:
            enc_hidden_dim = encoder.rnn.hidden_size * 2
            dec_hidden_dim = decoder.hidden_dim
            if enc_hidden_dim != dec_hidden_dim:
                self.bridge_h = nn.Linear(enc_hidden_dim, dec_hidden_dim)
                self.bridge_c = (
                    nn.Linear(enc_hidden_dim, dec_hidden_dim)
                    if encoder.rnn_type == "LSTM"
                    else None
                )
            else:
                self.bridge_h = None
                self.bridge_c = None
        else:
            self.bridge_h = None
            self.bridge_c = None

    def _bridge_hidden(self, hidden):
        if not self.encoder.rnn.bidirectional or self.bridge_h is None:
            return hidden

        if self.encoder.rnn_type == "LSTM":
            h, c = hidden
            n_layers = self.encoder.rnn.num_layers
            h_cat = torch.cat([h[0:n_layers], h[n_layers:]], dim=2)
            c_cat = torch.cat([c[0:n_layers], c[n_layers:]], dim=2)
            h_bridged = torch.tanh(self.bridge_h(h_cat))
            c_bridged = torch.tanh(self.bridge_c(c_cat))
            return (h_bridged, c_bridged)
        else:
            n_layers = self.encoder.rnn.num_layers
            h_cat = torch.cat([hidden[0:n_layers], hidden[n_layers:]], dim=2)
            return torch.tanh(self.bridge_h(h_cat))

    def forward(self, src, trg, teacher_forcing_ratio=0.4):
        encoder_outputs, hidden = self.encoder(src)
        hidden = self._bridge_hidden(hidden)
        outputs = self.decoder(
            trg, hidden, encoder_outputs, teacher_forcing_ratio=teacher_forcing_ratio
        )
        return outputs