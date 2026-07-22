import random
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        emb_dim_in = custom_emb_dim if custom_emb_dim else emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim_in)

        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(pretrained_emb, dtype=torch.float32)
            )
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
        # src: [batch_size, src_len]
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
        # hidden: [batch_size, hidden_dim]
        # encoder_outputs: [batch_size, src_len, enc_hidden_dim]
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

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.size(1)
        h_expanded = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.W_a(h_expanded) + self.U_a(encoder_outputs))
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
            self.embedding.weight.data.copy_(
                torch.tensor(pretrained_emb, dtype=torch.float32)
            )
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

    def forward_step(self, input_token, hidden, encoder_outputs):
        # input_token: [batch_size]
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))
        if self.project is not None:
            embedded = self.project(embedded)

        if self.attention_type != "none":
            h_top = (
                hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]
            )
            attn_weights = self.attention(h_top, encoder_outputs)
            context = torch.bmm(
                attn_weights.unsqueeze(1), encoder_outputs
            )  # [batch_size, 1, enc_hidden_dim]
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

        prediction = self.fc_out(fc_in.squeeze(1))  # [batch_size, vocab_size]
        return prediction, hidden, attn_weights

    def forward(
        self, trg, hidden, encoder_outputs, teacher_forcing_ratio=0.4
    ):
        batch_size, trg_len = trg.shape

        # FAST PATH: Fused cuDNN pass when teacher forcing triggers (tf_ratio ~ 0.4) on non-attention models
        use_teacher_forcing = self.training and (
            random.random() < teacher_forcing_ratio
        )

        if use_teacher_forcing and self.attention_type == "none":
            trg_in = trg[:, :-1]  # Slice inputs excluding last token
            embedded = self.dropout(self.embedding(trg_in))
            if self.project is not None:
                embedded = self.project(embedded)

            rnn_out, _ = self.rnn(embedded, hidden)  # Single fused cuDNN pass
            predictions = self.fc_out(
                rnn_out
            )  # [batch_size, trg_len-1, vocab_size]

            outputs = torch.zeros(
                batch_size, trg_len, self.vocab_size, device=trg.device
            )
            outputs[:, 1:] = predictions
            return outputs

        # STEP-BY-STEP PATH: For attention or when teacher forcing is skipped during training/inference
        outputs = torch.zeros(
            batch_size, trg_len, self.vocab_size, device=trg.device
        )
        input_token = trg[:, 0]  # <SOS> token

        for t in range(1, trg_len):
            pred, hidden, _ = self.forward_step(
                input_token, hidden, encoder_outputs
            )
            outputs[:, t] = pred
            teacher_force = self.training and (
                random.random() < teacher_forcing_ratio
            )
            top1 = pred.argmax(1)
            input_token = trg[:, t] if teacher_force else top1

        return outputs


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

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