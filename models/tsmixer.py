import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# TSMixer Block (FIXED)
# =========================
class TSMixerBlock(nn.Module):
    def __init__(
        self,
        seq_len,
        enc_in,
        ff_dim=2048,
        dropout=0.1,
        norm_type="L",
        activation="gelu"
    ):
        super().__init__()

        self.seq_len = seq_len
        self.enc_in = enc_in

        # =========================
        # Normalization (FEATURE dim)
        # =========================
        if norm_type == "L":
            self.norm1 = nn.LayerNorm(enc_in)
            self.norm2 = nn.LayerNorm(enc_in)
        else:
            self.norm1 = nn.BatchNorm1d(enc_in)
            self.norm2 = nn.BatchNorm1d(enc_in)

        self.dropout = nn.Dropout(dropout)

        # activation
        self.act = F.gelu if activation == "gelu" else F.relu

        # =========================
        # Temporal mixing (over time)
        # =========================
        self.temporal_fc = nn.Linear(seq_len, seq_len)

        # =========================
        # Feature mixing (over channels)
        # =========================
        self.fc1 = nn.Linear(enc_in, ff_dim)
        self.fc2 = nn.Linear(ff_dim, enc_in)

    def forward(self, x):
        # x: [B, L, C]
        residual = x

        # =========================
        # Temporal Mixing
        # =========================
        y = x.transpose(1, 2)      # [B, C, L]
        y = self.temporal_fc(y)    # mix time dimension
        y = y.transpose(1, 2)      # [B, L, C]

        x = self.norm1(residual + self.dropout(y))

        # =========================
        # Feature Mixing
        # =========================
        residual = x

        y = self.fc1(x)
        y = self.act(y)
        y = self.dropout(y)

        y = self.fc2(y)
        y = self.dropout(y)

        x = self.norm2(residual + y)

        return x


# =========================
# Full TSMixer Model
# =========================
class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        n_block=2,
        ff_dim=2048,
        dropout=0.1,
        norm_type="L",
        activation="gelu",
        target_slice=None
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.target_slice = target_slice

        # stack blocks
        self.blocks = nn.ModuleList([
            TSMixerBlock(
                seq_len=seq_len,
                enc_in=enc_in,
                ff_dim=ff_dim,
                dropout=dropout,
                norm_type=norm_type,
                activation=activation
            )
            for _ in range(n_block)
        ])

        # temporal projection (L → pred_len)
        self.proj = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, L, C]

        for block in self.blocks:
            x = block(x)

        # optional feature selection
        if self.target_slice is not None:
            x = x[:, :, self.target_slice]

        # [B, L, C] → [B, C, L]
        x = x.permute(0, 2, 1)

        # project time dimension
        x = self.proj(x)

        # [B, C, pred_len] → [B, pred_len, C]
        x = x.permute(0, 2, 1)

        return x
