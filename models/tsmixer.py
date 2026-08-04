# coding=utf-8
"""
PyTorch implementation of TSMixer
Compatible with PatchMixer / PyTorch training pipeline.
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    TSMixer residual block.

    Input:
        [B, L, C]

    Output:
        [B, L, C]
    """

    def __init__(
        self,
        seq_len,
        n_channels,
        norm_type='L',
        activation='relu',
        dropout=0.2,
        ff_dim=2048
    ):
        super().__init__()

        # -------------------------------------------------
        # Normalization
        # -------------------------------------------------
        if norm_type == 'L':
            self.norm1 = nn.LayerNorm(n_channels)
            self.norm2 = nn.LayerNorm(n_channels)
        else:
            self.norm1 = nn.BatchNorm1d(n_channels)
            self.norm2 = nn.BatchNorm1d(n_channels)

        # -------------------------------------------------
        # Activation
        # -------------------------------------------------
        if activation.lower() == 'relu':
            act = nn.ReLU()
        elif activation.lower() == 'gelu':
            act = nn.GELU()
        elif activation.lower() == 'silu':
            act = nn.SiLU()
        else:
            act = nn.ReLU()

        # -------------------------------------------------
        # Temporal Linear
        #
        # [B, L, C]
        # -> [B, C, L]
        # -> Linear(L, L)
        # -> [B, L, C]
        # -------------------------------------------------
        self.temporal_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            act,
            nn.Dropout(dropout)
        )

        # -------------------------------------------------
        # Feature Linear
        # -------------------------------------------------
        self.feature_mlp = nn.Sequential(
            nn.Linear(n_channels, ff_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(ff_dim, n_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        # =================================================
        # Temporal Mixing
        # =================================================

        residual = x

        x = self.norm1(x)

        # [B, L, C] -> [B, C, L]
        x = x.transpose(1, 2)

        # Temporal MLP
        x = self.temporal_mlp(x)

        # [B, C, L] -> [B, L, C]
        x = x.transpose(1, 2)

        x = x + residual

        # =================================================
        # Feature Mixing
        # =================================================

        residual = x

        x = self.norm2(x)

        x = self.feature_mlp(x)

        x = x + residual

        return x


class Model(nn.Module):
    """
    TSMixer model compatible with PatchMixer Exp_Main.

    Input:
        [B, seq_len, enc_in]

    Output:
        [B, pred_len, enc_in]
    """

    def __init__(self, args):
        super(Model, self).__init__()

        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.enc_in = args.enc_in

        self.norm_type = args.norm_type
        self.activation = args.activation
        self.n_block = args.n_block
        self.dropout = args.dropout
        self.ff_dim = args.ff_dim

        # -------------------------------------------------
        # TSMixer blocks
        # -------------------------------------------------

        self.blocks = nn.ModuleList([
            ResBlock(
                seq_len=self.seq_len,
                n_channels=self.enc_in,
                norm_type=self.norm_type,
                activation=self.activation,
                dropout=self.dropout,
                ff_dim=self.ff_dim
            )
            for _ in range(self.n_block)
        ])

        # -------------------------------------------------
        # Forecasting head
        #
        # [B, L, C]
        # -> [B, C, L]
        # -> Dense(L -> pred_len)
        # -> [B, pred_len, C]
        # -------------------------------------------------

        self.forecast = nn.Linear(
            self.seq_len,
            self.pred_len
        )

    def forward(self, x):

        # x:
        # [B, seq_len, enc_in]

        for block in self.blocks:
            x = block(x)

        # [B, L, C] -> [B, C, L]
        x = x.transpose(1, 2)

        # [B, C, L] -> [B, C, pred_len]
        x = self.forecast(x)

        # [B, C, pred_len] -> [B, pred_len, C]
        x = x.transpose(1, 2)

        return x
