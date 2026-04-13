# coding=utf-8
"""PyTorch implementation of TSMixer (compatible with Exp_Basic)."""

import torch
import torch.nn as nn


# =========================
# Residual Block
# =========================
class TSMixerBlock(nn.Module):
    def __init__(self, seq_len, hidden_dim, norm_type='L', activation='gelu',
                 dropout=0.05, ff_dim=2048):
        super().__init__()

        self.norm_type = norm_type

        # normalization
        if norm_type == 'L':
            self.norm1 = nn.LayerNorm(seq_len)
            self.norm2 = nn.LayerNorm(seq_len)
        else:
            self.norm1 = nn.BatchNorm1d(seq_len)
            self.norm2 = nn.BatchNorm1d(seq_len)

        self.act = getattr(torch.nn.functional, activation)

        self.dropout = nn.Dropout(dropout)

        # Temporal mixing
        self.temporal_fc = nn.Linear(seq_len, seq_len)

        # Feature mixing
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)

    def forward(self, x):
        # x: [B, L, C]

        residual = x

        # ===== Temporal mixing =====
        y = x.transpose(1, 2)  # [B, C, L]
        y = self.temporal_fc(y)
        y = y.transpose(1, 2)  # [B, L, C]
        y = self.dropout(y)

        x = residual + y

        # ===== Feature mixing =====
        residual = x

        y = self.fc1(x)
        y = torch.relu(y) if self.act == torch.nn.functional.relu else y
        y = self.dropout(y)

        y = self.fc2(y)
        y = self.dropout(y)

        return residual + y


# =========================
# Model
# =========================
class Model(nn.Module):
    def __init__(
        self,
        input_shape,
        pred_len,
        norm_type='L',
        activation='gelu',
        n_block=2,
        dropout=0.05,
        ff_dim=2048,
        target_slice=None,
    ):
        super().__init__()

        seq_len, enc_in = input_shape
        self.pred_len = pred_len
        self.target_slice = target_slice

        self.blocks = nn.ModuleList([
            TSMixerBlock(
                seq_len=seq_len,
                hidden_dim=enc_in,
                norm_type=norm_type,
                activation=activation,
                dropout=dropout,
                ff_dim=ff_dim
            )
            for _ in range(n_block)
        ])

        # final projection
        self.proj = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, L, C]

        for block in self.blocks:
            x = block(x)

        # feature selection
        if self.target_slice is not None:
            x = x[:, :, self.target_slice]

        # [B, L, C] -> [B, C, L]
        x = x.permute(0, 2, 1)

        # predict future
        x = self.proj(x)

        # [B, C, pred_len] -> [B, pred_len, C]
        x = x.permute(0, 2, 1)

        return x
