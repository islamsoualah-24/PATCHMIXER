import torch
import torch.nn as nn


# =========================
# TSMixer Block (Fixed)
# =========================
class TSMixerBlock(nn.Module):
    def __init__(
        self,
        seq_len,
        num_features,
        ff_dim=2048,
        dropout=0.1,
        norm_type="L",
        activation="gelu"
    ):
        super().__init__()

        self.seq_len = seq_len
        self.num_features = num_features

        # normalization
        if norm_type == "L":
            self.norm1 = nn.LayerNorm(num_features)
            self.norm2 = nn.LayerNorm(num_features)
        else:
            self.norm1 = nn.BatchNorm1d(num_features)
            self.norm2 = nn.BatchNorm1d(num_features)

        # activation
        self.act = getattr(torch.nn.functional, activation)

        self.dropout = nn.Dropout(dropout)

        # Temporal mixing (mix along time axis)
        self.temporal_fc = nn.Linear(seq_len, seq_len)

        # Feature mixing (MLP per time step)
        self.fc1 = nn.Linear(num_features, ff_dim)
        self.fc2 = nn.Linear(ff_dim, num_features)

    def forward(self, x):
        # x: [B, L, C]
        residual = x

        # =========================
        # Temporal Mixing
        # =========================
        y = x.transpose(1, 2)              # [B, C, L]
        y = self.temporal_fc(y)            # mix time dimension
        y = y.transpose(1, 2)              # [B, L, C]
        y = self.dropout(y)

        x = residual + y

        # optional norm (stable training)
        if isinstance(self.norm1, nn.LayerNorm):
            x = self.norm1(x)
        else:
            x = x.transpose(1, 2)
            x = self.norm1(x)
            x = x.transpose(1, 2)

        # =========================
        # Feature Mixing
        # =========================
        residual = x

        y = self.fc1(x)
        y = self.act(y)
        y = self.dropout(y)

        y = self.fc2(y)
        y = self.dropout(y)

        x = residual + y

        # optional norm
        if isinstance(self.norm2, nn.LayerNorm):
            x = self.norm2(x)
        else:
            x = x.transpose(1, 2)
            x = self.norm2(x)
            x = x.transpose(1, 2)

        return x


# =========================
# Full TSMixer Model (Fixed)
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
        self.target_slice = target_slice

        self.blocks = nn.ModuleList([
            TSMixerBlock(
                seq_len=seq_len,
                num_features=enc_in,
                ff_dim=ff_dim,
                dropout=dropout,
                norm_type=norm_type,
                activation=activation
            )
            for _ in range(n_block)
        ])

        # projection: map time -> prediction horizon
        self.proj = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, L, C]

        for block in self.blocks:
            x = block(x)

        # optional feature selection
        if self.target_slice is not None:
            x = x[:, :, self.target_slice]

        # [B, L, C] -> [B, C, L]
        x = x.permute(0, 2, 1)

        # temporal projection
        x = self.proj(x)

        # [B, C, pred_len] -> [B, pred_len, C]
        x = x.permute(0, 2, 1)

        return x
