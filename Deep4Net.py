
import math
import types
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

# ------------------------------
# Hyperparameters (default set)
# ------------------------------
HP_DEFAULT2 = types.SimpleNamespace(
    in_chans=2,
    epoch_time=60,
    sampling_rate=125,
    n_signal=1,
    n_classes=2,
    n_filters_time=40,
    n_filters_spat=40,
    filter_time_length=1,
    pool_time_length=0.04,
    pool_time_stride=0.04,
    n_filters2=30,
    filter_length2=20,
    n_filters3=40,
    filter_length3=10,
    n_filters4=50,
    filter_length4=5,
    hidden=32,
    drop_prob=0.5,
    init_xavier=0,
)

# ------------------------------
# Attention utilities
# ------------------------------
class CrossAttention(nn.Module):
    """
    Simple cross-attention with scalar embedding (embed_dim=1).

    Inputs:
        query, key, value: (B, T, E)

    Returns:
        output: (B, T, E)
        attn_weights: (B, T, T)
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer   = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)
        self.softmax     = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        q = self.query_layer(query)
        k = self.key_layer(key)
        v = self.value_layer(value)
        attn_scores  = torch.matmul(q, k.transpose(-2, -1))   # (B, T, T)
        attn_weights = self.softmax(attn_scores)
        out = torch.matmul(attn_weights, v)                   # (B, T, E)
        return out, attn_weights


class SelfAttBlock(nn.Module):
    """Transformer-style self-attention block (Batch-first)."""
    def __init__(self, d_model=32, nhead=1, dropout=0.1, ffn_hidden=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1  = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, ffn_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model)
        )
        self.ln2  = nn.LayerNorm(d_model)

    def forward(self, x):              # x: (B, T, d_model)
        a,_ = self.attn(x, x, x, need_weights=False)
        x   = self.ln1(x + a)
        y   = self.ffn(x)
        x   = self.ln2(x + y)
        return x


class SinePositionalEncoding(nn.Module):
    """Standard sine/cosine positional encoding added to the token sequence."""
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x):  # x: (B, L, d_model)
        L = x.size(1)
        return x + self.pe[:, :L, :]


# ------------------------------
# Model 1: Deep4AttNet (cross-attention between ears)
# ------------------------------
class Deep4AttNet(nn.Module):
    """
    Two 1D encoders (left/right ear) + pointwise temporal compression + cross-attention (L->R and R->L).

    Expected input: x of shape (B, n_signal=1, in_chans=2, T).
    """
    def __init__(self, hp: types.SimpleNamespace = None):
        super().__init__()
        # Resolve hyperparameters (fix: always use self.hp)
        self.hp = hp if hp is not None else HP_DEFAULT2
        hp = self.hp

        # Shapes
        self.in_chans           = hp.in_chans
        self.n_signal           = hp.n_signal
        self.input_time_length  = int(hp.epoch_time * hp.sampling_rate)

        # Encoder params
        self.n_filters_time     = hp.n_filters_time
        self.filter_time_length = round(hp.filter_time_length * hp.sampling_rate)
        self.n_filters_spat     = hp.n_filters_spat
        self.pool_time_length   = round(hp.pool_time_length * hp.sampling_rate)
        self.pool_time_stride   = round(hp.pool_time_stride * hp.sampling_rate)
        self.n_filters2         = hp.n_filters2
        self.filter_length2     = hp.filter_length2
        self.n_filters3         = hp.n_filters3
        self.filter_length3     = hp.filter_length3
        self.n_filters4         = hp.n_filters4
        self.filter_length4     = hp.filter_length4

        # Classifier params
        self.hidden    = hp.hidden
        self.drop_prob = hp.drop_prob
        self.n_classes = hp.n_classes

        # Dummy tensor for computing encoded length
        self.input_foo = torch.zeros([32, self.n_signal, 1, self.input_time_length])

        # Encoders (left/right)
        self.encoder_l = nn.Sequential(
            nn.Conv2d(self.n_signal, self.n_filters_time, (1, self.filter_time_length), stride=1, bias=True),
            nn.BatchNorm2d(self.n_filters_time, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride)),

            nn.Conv2d(self.n_filters_time, self.n_filters2, (1, self.filter_length2), stride=1, bias=False),
            nn.BatchNorm2d(self.n_filters2, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride)),

            nn.Conv2d(self.n_filters2, self.n_filters3, (1, self.filter_length3), stride=1, bias=False),
            nn.BatchNorm2d(self.n_filters3, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride)),

            # Final pointwise conv to compress channels to 1
            nn.Conv2d(self.n_filters3, 1, (1, self.filter_length4), stride=1, bias=False),
            nn.BatchNorm2d(1, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
        )
        self.encoder_r = nn.Sequential(
            nn.Conv2d(self.n_signal, self.n_filters_time, (1, self.filter_time_length), stride=1, bias=True),
            nn.BatchNorm2d(self.n_filters_time, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride)),

            nn.Conv2d(self.n_filters_time, self.n_filters2, (1, self.filter_length2), stride=1, bias=False),
            nn.BatchNorm2d(self.n_filters2, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride)),

            nn.Conv2d(self.n_filters2, self.n_filters3, (1, self.filter_length3), stride=1, bias=False),
            nn.BatchNorm2d(self.n_filters3, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride)),

            nn.Conv2d(self.n_filters3, 1, (1, self.filter_length4), stride=1, bias=False),
            nn.BatchNorm2d(1, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
        )

        # Token length (T_enc)
        self.len_encoding = self._len_encoding()

        # Cross-attention (directed, L->R and R->L), embedding dim = 1
        self.cross_Att_lr = CrossAttention(embed_dim=1)
        self.cross_Att_rl = CrossAttention(embed_dim=1)

        # Classifier takes concatenated attended tokens (2 * T_enc)
        self.classifier = nn.Sequential(
            nn.Linear(self.len_encoding * 2, self.hidden, bias=False),
            nn.BatchNorm1d(self.hidden, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.hidden, self.n_classes)
        )

        # Optional: Xavier initialization
        if getattr(hp, 'init_xavier', 0):
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    if getattr(m, 'bias', None) is not None and m.bias is not False:
                        nn.init.zeros_(m.bias)

    def _len_encoding(self):
        self.eval()
        with torch.no_grad():
            out = self.encoder_l(self.input_foo)  # (32,1,1,T_enc)
            out = out.view(out.size(0), -1)       # (32, T_enc)
        return out.shape[-1]

    def forward(self, x):
        """
        x: (B, n_signal=1, in_chans=2, T)
        """
        # Split into left/right single-channel inputs
        xl = x[:, :, 0, :].unsqueeze(1)  # (B,1,1,T)
        xr = x[:, :, 1, :].unsqueeze(1)  # (B,1,1,T)

        # Encode
        out_l = self.encoder_l(xl).view(x.size(0), -1)  # (B, T_enc)
        out_r = self.encoder_r(xr).view(x.size(0), -1)  # (B, T_enc)
        assert out_l.shape[1] == self.len_encoding and out_r.shape[1] == self.len_encoding

        # Convert to tokens (E=1) and apply cross-attention
        q_l = out_l.unsqueeze(-1)                       # (B, T_enc, 1)
        k_r = out_r.unsqueeze(-1)                       # (B, T_enc, 1)
        out_lr, att_lr = self.cross_Att_lr(q_l, k_r, k_r)

        q_r = out_r.unsqueeze(-1)                       # (B, T_enc, 1)
        k_l = out_l.unsqueeze(-1)                       # (B, T_enc, 1)
        out_rl, att_rl = self.cross_Att_rl(q_r, k_l, k_l)

        # (Optional) store attention maps for analysis
        self.last_att_lr = att_lr
        self.last_att_rl = att_rl

        # Classifier input
        feat = torch.cat([out_lr.squeeze(-1), out_rl.squeeze(-1)], dim=1)  # (B, 2*T_enc)
        y = self.classifier(feat)
        return y


# ------------------------------
# Model 2: Deep4SelfAttNet (per-ear self-attention, concatenated)
# ------------------------------
class Deep4SelfAttNet(nn.Module):
    """
    Remove cross-attention and apply self-attention within each ear channel.
    Concatenate left/right embeddings for classification.
    """
    def __init__(self, hp: types.SimpleNamespace = None, d_model=32, nhead=1, nlayers=1):
        super().__init__()
        self.hp = hp if hp is not None else HP_DEFAULT2
        hp = self.hp

        # Encoder hyperparameters
        self.in_chans = hp.in_chans
        self.n_signal = hp.n_signal
        self.input_time_length = int(hp.epoch_time * hp.sampling_rate)
        self.n_filters_time = hp.n_filters_time
        self.filter_time_length = round(hp.filter_time_length * hp.sampling_rate)
        self.pool_time_length   = round(hp.pool_time_length   * hp.sampling_rate)
        self.pool_time_stride   = round(hp.pool_time_stride   * hp.sampling_rate)
        self.n_filters2, self.filter_length2 = hp.n_filters2, hp.filter_length2
        self.n_filters3, self.filter_length3 = hp.n_filters3, hp.filter_length3
        self.n_filters4, self.filter_length4 = hp.n_filters4, hp.filter_length4
        self.hidden, self.drop_prob, self.n_classes = hp.hidden, hp.drop_prob, hp.n_classes

        # Dummy for token length
        self.input_foo = torch.zeros([32, self.n_signal, 1, self.input_time_length])

        def make_encoder():
            return nn.Sequential(
                nn.Conv2d(self.n_signal, self.n_filters_time, (1, self.filter_time_length), stride=1, bias=True),
                nn.BatchNorm2d(self.n_filters_time, momentum=0.1, affine=True, eps=1e-5),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, max(1, self.pool_time_length)), stride=(1, max(1, self.pool_time_stride))),

                nn.Conv2d(self.n_filters_time, self.n_filters2, (1, self.filter_length2), stride=1, bias=False),
                nn.BatchNorm2d(self.n_filters2, momentum=0.1, affine=True, eps=1e-5),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, max(1, self.pool_time_length)), stride=(1, max(1, self.pool_time_stride))),

                nn.Conv2d(self.n_filters2, self.n_filters3, (1, self.filter_length3), stride=1, bias=False),
                nn.BatchNorm2d(self.n_filters3, momentum=0.1, affine=True, eps=1e-5),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, max(1, self.pool_time_length)), stride=(1, max(1, self.pool_time_stride))),

                nn.Conv2d(self.n_filters3, self.n_filters4, (1, self.filter_length4), stride=1, bias=False),
                nn.BatchNorm2d(self.n_filters4, momentum=0.1, affine=True, eps=1e-5),
                nn.ELU(),
                nn.Conv2d(self.n_filters4, 1, (1, 1), stride=1, bias=False),  # (B,1,1,T')
            )

        self.encoder_l = make_encoder()
        self.encoder_r = make_encoder()

        # Token length
        self.len_tokens = self._len_encoding()      # typically ~34
        # Token projection per ear
        self.proj = nn.Linear(1, d_model)           # (B, T', 1) -> (B, T', d_model)
        self.sa_blocks = nn.ModuleList([
            SelfAttBlock(d_model, nhead, self.drop_prob, ffn_hidden=4*d_model)
            for _ in range(nlayers)
        ])

        # Classifier on concatenated ear embeddings
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, self.hidden, bias=False),
            nn.BatchNorm1d(self.hidden, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), nn.Dropout(self.drop_prob),
            nn.Linear(self.hidden, self.n_classes)
        )

    def _len_encoding(self):
        self.eval()
        out = self.encoder_l(self.input_foo)           # (32,1,1,T')
        out = out.view(out.size(0), -1)                # (32, T')
        return out.shape[-1]

    def _encode_tokens(self, encoder: nn.Module, x_one: torch.Tensor) -> torch.Tensor:
        """
        x_one: (B, 1, 1, T) -> returns pooled embedding (B, d_model)
        """
        B = x_one.size(0)
        out = encoder(x_one).view(B, self.len_tokens, 1)  # (B, T', 1)
        z = self.proj(out)                                # (B, T', d_model)
        for blk in self.sa_blocks:
            z = blk(z)                                    # (B, T', d_model)
        z = z.mean(dim=1, keepdim=False)                  # (B, d_model)
        return z

    def forward(self, x):
        # Split channels
        xl = x[:, :, 0, :].unsqueeze(1)   # (B,1,1,T)
        xr = x[:, :, 1, :].unsqueeze(1)   # (B,1,1,T)
        zl = self._encode_tokens(self.encoder_l, xl)  # (B, d_model)
        zr = self._encode_tokens(self.encoder_r, xr)  # (B, d_model)
        feat = torch.cat([zl, zr], dim=1)             # (B, 2*d_model)
        y = self.classifier(feat)
        return y


# ------------------------------
# Model 3: Deep4LiteTransNet (lightweight Transformer over concatenated ear tokens)
# ------------------------------
class Deep4LiteTransNet(nn.Module):
    """
    Concatenate left/right token sequences and process with a lightweight Transformer.

    Expected input: x of shape (B, n_signal=1, in_chans=2, T).
    """
    def __init__(self, hp: types.SimpleNamespace = None, d_model=48, nhead=2, nlayers=1):
        super().__init__()
        self.hp = hp if hp is not None else HP_DEFAULT2
        hp = self.hp

        # Encoder hyperparameters
        self.in_chans = hp.in_chans
        self.n_signal = hp.n_signal
        self.input_time_length = int(hp.epoch_time * hp.sampling_rate)
        self.n_filters_time = hp.n_filters_time
        self.filter_time_length = round(hp.filter_time_length * hp.sampling_rate)
        self.pool_time_length   = round(hp.pool_time_length   * hp.sampling_rate)
        self.pool_time_stride   = round(hp.pool_time_stride   * hp.sampling_rate)
        self.n_filters2, self.filter_length2 = hp.n_filters2, hp.filter_length2
        self.n_filters3, self.filter_length3 = hp.n_filters3, hp.filter_length3
        self.n_filters4, self.filter_length4 = hp.n_filters4, hp.filter_length4
        self.hidden, self.drop_prob, self.n_classes = hp.hidden, hp.drop_prob, hp.n_classes

        self.input_foo = torch.zeros([32, self.n_signal, 1, self.input_time_length])

        def make_encoder():
            return nn.Sequential(
                nn.Conv2d(self.n_signal, self.n_filters_time, (1, self.filter_time_length), stride=1, bias=True),
                nn.BatchNorm2d(self.n_filters_time, momentum=0.1, affine=True, eps=1e-5),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, max(1, self.pool_time_length)), stride=(1, max(1, self.pool_time_stride))),
                nn.Conv2d(self.n_filters_time, self.n_filters2, (1, self.filter_length2), stride=1, bias=False),
                nn.BatchNorm2d(self.n_filters2, momentum=0.1, affine=True, eps=1e-5),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, max(1, self.pool_time_length)), stride=(1, max(1, self.pool_time_stride))),
                nn.Conv2d(self.n_filters2, self.n_filters3, (1, self.filter_length3), stride=1, bias=False),
                nn.BatchNorm2d(self.n_filters3, momentum=0.1, affine=True, eps=1e-5),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, max(1, self.pool_time_length)), stride=(1, max(1, self.pool_time_stride))),
                nn.Conv2d(self.n_filters3, self.n_filters4, (1, self.filter_length4), stride=1, bias=False),
                nn.BatchNorm2d(self.n_filters4, momentum=0.1, affine=True, eps=1e-5),
                nn.ELU(),
                nn.Conv2d(self.n_filters4, 1, (1, 1), stride=1, bias=False),
            )
        self.encoder_l = make_encoder()
        self.encoder_r = make_encoder()

        # Tokenization
        self.len_tokens = self._len_encoding()          # T'
        self.proj       = nn.Linear(1, d_model)         # (B, T', 1) -> (B, T', d_model)
        self.pos        = SinePositionalEncoding(d_model, max_len=2048)
        self.type_emb   = nn.Embedding(2, d_model)      # 0: left, 1: right

        # Lightweight Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4 * d_model, dropout=self.drop_prob,
            batch_first=True, activation='gelu', norm_first=True
        )
        self.tr_encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, self.hidden, bias=False),
            nn.BatchNorm1d(self.hidden, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), nn.Dropout(self.drop_prob),
            nn.Linear(self.hidden, self.n_classes)
        )

    def _len_encoding(self):
        self.eval()
        with torch.no_grad():
            out = self.encoder_l(self.input_foo).view(self.input_foo.size(0), -1)  # (32, T')
        return out.shape[-1]

    def _to_tokens(self, enc, x_one):                   # enc: encoder_l/r, x_one: (B,1,1,T)
        z = enc(x_one).view(x_one.size(0), self.len_tokens, 1)  # (B, T', 1)
        return self.proj(z)                                      # (B, T', d_model)

    def forward(self, x):
        xl = x[:, :, 0, :].unsqueeze(1)                 # (B,1,1,T)
        xr = x[:, :, 1, :].unsqueeze(1)

        tl = self._to_tokens(self.encoder_l, xl)        # (B, T', d_model)
        tr = self._to_tokens(self.encoder_r, xr)        # (B, T', d_model)

        # Add channel-type embedding and position
        B, T, D = tl.size()
        tl = tl + self.type_emb.weight[0].view(1, 1, D)
        tr = tr + self.type_emb.weight[1].view(1, 1, D)
        seq = torch.cat([tl, tr], dim=1)                # (B, 2*T', D)
        seq = self.pos(seq)

        h = self.tr_encoder(seq)                        # (B, 2*T', D)
        h = h.mean(dim=1)                               # global average pooling
        y = self.classifier(h)
        return y



if __name__ == "__main__":
    pass
