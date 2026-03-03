import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from layers.PatchTST_layers import *
from layers.RevIN import RevIN

# =========================================================
# 🔥 Temporal ASPP Module (Multi-Scale Dilated Conv1D)
# =========================================================

class TemporalASPP(nn.Module):
    def __init__(self, dim, kernel_size=3, dilations=[1, 2, 4, 8]):
        super().__init__()
        
        self.branches = nn.ModuleList()
        
        for d in dilations:
            self.branches.append(
                nn.Conv1d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=kernel_size,
                    padding='same',
                    dilation=d,
                    groups=dim   # depthwise convolution
                )
            )
        
        self.project = nn.Sequential(
            nn.Conv1d(dim * len(dilations), dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        # x: [B*N, patch_num, d_model]
        outputs = []
        for conv in self.branches:
            outputs.append(conv(x))
        
        x = torch.cat(outputs, dim=1)  # concatenate along channel dimension
        x = self.project(x)
        return x


# =========================================================
# 🔥 PatchMixer Layer with ASPP
# =========================================================

class PatchMixerLayer(nn.Module):
    def __init__(self, dim, a, kernel_size=3):
        super().__init__()
        
        # Multi-Scale Temporal Mixing
        self.ASPP = TemporalASPP(dim=dim, kernel_size=kernel_size)

        # Channel Mixing (1x1 Conv)
        self.Conv_1x1 = nn.Sequential(
            nn.Conv1d(dim, a, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(a)
        )

    def forward(self, x):
        # x: [B*N, patch_num, d_model]
        
        # Residual Multi-Scale Mixing
        x = x + self.ASPP(x)

        # Channel Mixing
        x = self.Conv_1x1(x)

        return x


# =========================================================
# 🔥 Main Model Wrapper
# =========================================================

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = Backbone(configs)

    def forward(self, x):
        return self.model(x)


# =========================================================
# 🔥 Backbone
# =========================================================

class Backbone(nn.Module):
    def __init__(self, configs, revin=True, affine=True, subtract_last=False):
        super().__init__()

        self.nvals = configs.enc_in
        self.lookback = configs.seq_len
        self.forecasting = configs.pred_len
        self.patch_size = configs.patch_len
        self.stride = configs.stride
        self.kernel_size = configs.mixer_kernel_size

        self.depth = configs.e_layers
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.head_dropout = configs.head_dropout

        # Padding
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))

        # Number of patches
        self.patch_num = int((self.lookback - self.patch_size) / self.stride + 1) + 1
        self.a = self.patch_num

        # Patch Embedding
        self.W_P = nn.Linear(self.patch_size, self.d_model)

        # PatchMixer Blocks (with ASPP)
        self.PatchMixer_blocks = nn.ModuleList()
        for _ in range(self.depth):
            self.PatchMixer_blocks.append(
                PatchMixerLayer(
                    dim=self.patch_num,
                    a=self.a,
                    kernel_size=self.kernel_size
                )
            )

        # Prediction Heads
        self.head0 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.patch_num * self.d_model, self.forecasting),
            nn.Dropout(self.head_dropout)
        )

        self.head1 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.a * self.d_model, int(self.forecasting * 2)),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(int(self.forecasting * 2), self.forecasting),
            nn.Dropout(self.head_dropout)
        )

        self.dropout_layer = nn.Dropout(self.dropout)

        # RevIN
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(self.nvals, affine=affine, subtract_last=subtract_last)

    def forward(self, x):

        bs = x.shape[0]
        nvars = x.shape[-1]

        # RevIN normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')

        # [B, seq_len, nvars] → [B, nvars, seq_len]
        x = x.permute(0, 2, 1)

        # Padding
        x = self.padding_patch_layer(x)

        # Create patches
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # shape: [B, nvars, patch_num, patch_size]

        # Patch embedding
        x = self.W_P(x)
        # shape: [B, nvars, patch_num, d_model]

        # Merge batch & variables
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # shape: [B*nvars, patch_num, d_model]

        x = self.dropout_layer(x)

        # First prediction head
        u = self.head0(x)

        # PatchMixer Blocks
        for block in self.PatchMixer_blocks:
            x = block(x)

        # Second prediction head
        x = self.head1(x)

        # Residual forecast
        x = u + x

        # Reshape back
        x = torch.reshape(x, (bs, nvars, -1))
        x = x.permute(0, 2, 1)

        # Denormalization
        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x
