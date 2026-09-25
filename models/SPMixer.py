__all__ = ['Model']

import torch
from torch import nn
import torch.nn.functional as F

from layers.PatchTST_layers import *
from layers.RevIN import RevIN


# ============================================================
# 1. PatchMixer Block
# ============================================================

class PatchMixerLayer(nn.Module):
    """
    Original PatchMixer-style temporal mixing block.

    Input:
        [B*N, PatchNum, D]

    Output:
        [B*N, PatchNum, D]
    """

    def __init__(self, dim, a, kernel_size=8, dropout=0.0):
        super().__init__()

        self.Resnet = nn.Sequential(
            nn.Conv1d(
                dim,
                dim,
                kernel_size=kernel_size,
                groups=dim,
                padding='same'
            ),
            nn.GELU(),
            nn.BatchNorm1d(dim)
        )

        self.Conv_1x1 = nn.Sequential(
            nn.Conv1d(dim, a, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(a)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # x: [B*N, PatchNum, D]

        x = x + self.Resnet(x)

        x = self.Conv_1x1(x)

        x = self.dropout(x)

        return x


# ============================================================
# 2. Multi-Scale PatchMixer Branch
# ============================================================

class MultiScalePatchMixer(nn.Module):
    """
    Multi-scale PatchMixer.

    Uses multiple patch sizes:
        P8
        P16
        P32

    Each scale has its own patch embedding and PatchMixer blocks.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        d_model,
        depth,
        dropout,
        head_dropout,
        kernel_size,
        patch_sizes=(8, 16, 32),
        strides=(4, 8, 16)
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        self.patch_sizes = patch_sizes
        self.strides = strides

        self.branches = nn.ModuleList()

        for patch_size, stride in zip(
            patch_sizes,
            strides
        ):

            # Same padding strategy as original PatchMixer
            patch_num = (
                int((seq_len - patch_size) / stride + 1)
                + 1
            )

            a = patch_num

            branch = nn.ModuleDict({

                "padding":
                    nn.ReplicationPad1d((0, stride)),

                "embedding":
                    nn.Linear(
                        patch_size,
                        d_model
                    ),

                "blocks":
                    nn.ModuleList([
                        PatchMixerLayer(
                            dim=patch_num,
                            a=a,
                            kernel_size=kernel_size,
                            dropout=dropout
                        )
                        for _ in range(depth)
                    ]),

                "head":
                    nn.Sequential(
                        nn.Flatten(start_dim=-2),

                        nn.Linear(
                            a * d_model,
                            pred_len * 2
                        ),

                        nn.GELU(),

                        nn.Dropout(head_dropout),

                        nn.Linear(
                            pred_len * 2,
                            pred_len
                        ),

                        nn.Dropout(head_dropout)
                    )
            })

            self.branches.append(branch)

        # Fusion of P8/P16/P32
        self.fusion = nn.Sequential(
            nn.Linear(
                len(patch_sizes) * pred_len,
                pred_len
            ),
            nn.GELU(),
            nn.Dropout(head_dropout)
        )

    def forward(self, x):
        """
        x:
            [B, N, L]

        returns:
            [B, N, H]
        """

        outputs = []

        for branch in self.branches:

            # [B, N, L]
            z = branch["padding"](x)

            # [B, N, PatchNum, PatchSize]
            z = z.unfold(
                dimension=-1,
                size=branch["embedding"].in_features,
                step=self.strides[
                    len(outputs)
                ]
            )

            # [B, N, PatchNum, D]
            z = branch["embedding"](z)

            B, N, P, D = z.shape

            # Channel independent processing
            z = z.reshape(
                B * N,
                P,
                D
            )

            for block in branch["blocks"]:
                z = block(z)

            # [B*N, H]
            z = branch["head"](z)

            # [B, N, H]
            z = z.reshape(
                B,
                N,
                self.pred_len
            )

            outputs.append(z)

        # [B, N, H]
        fused = torch.cat(
            outputs,
            dim=-1
        )

        fused = self.fusion(fused)

        return fused


# ============================================================
# 3. Trend Extraction
# ============================================================

class MovingAverage(nn.Module):

    def __init__(self, kernel_size):
        super().__init__()

        self.kernel_size = kernel_size

    def forward(self, x):
        """
        x:
            [B, L, N]
        """

        k = self.kernel_size

        # Make sure kernel is odd
        if k % 2 == 0:
            k += 1

        # [B, N, L]
        x = x.permute(0, 2, 1)

        # Replication padding
        pad = k // 2

        x = F.pad(
            x,
            (pad, pad),
            mode='replicate'
        )

        # Moving average
        trend = F.avg_pool1d(
            x,
            kernel_size=k,
            stride=1
        )

        # [B, L, N]
        trend = trend.permute(
            0, 2, 1
        )

        return trend


class SeriesDecomposition(nn.Module):

    def __init__(self, kernel_size):
        super().__init__()

        self.moving_avg = MovingAverage(
            kernel_size
        )

    def forward(self, x):

        trend = self.moving_avg(x)

        residual = x - trend

        return trend, residual


# ============================================================
# 4. Linear Trend Branch
# ============================================================

class LinearTrendBranch(nn.Module):
    """
    Linear forecasting branch.

    Similar motivation to NLinear/DLinear:
    explicitly model the long-term trend.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        nvars,
        dropout=0.0
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len

        # Shared temporal projection
        self.linear = nn.Linear(
            seq_len,
            pred_len
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, trend):

        # trend:
        # [B, L, N]

        x = trend.permute(
            0, 2, 1
        )

        # [B, N, H]
        x = self.linear(x)

        x = self.dropout(x)

        return x


# ============================================================
# 5. Adaptive Channel Gate
# ============================================================

class AdaptiveChannelGate(nn.Module):
    """
    Learnable channel mixing gate.

    Important:
    PatchMixer remains channel-independent by default.

    This module only introduces a gated residual
    channel interaction.
    """

    def __init__(
        self,
        nvars,
        pred_len,
        hidden_dim=64,
        dropout=0.0
    ):
        super().__init__()

        self.nvars = nvars
        self.pred_len = pred_len

        # Channel interaction
        self.channel_mixer = nn.Sequential(

            nn.Conv1d(
                nvars,
                nvars,
                kernel_size=1
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Conv1d(
                nvars,
                nvars,
                kernel_size=1
            )
        )

        # Learnable gate
        self.gate = nn.Sequential(

            nn.Linear(
                nvars,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                nvars
            ),

            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x:
            [B, N, H]
        """

        # Channel statistics
        context = x.mean(
            dim=-1
        )

        # [B, N]
        gate = self.gate(context)

        # [B, N, 1]
        gate = gate.unsqueeze(-1)

        # Channel mixing
        mixed = self.channel_mixer(x)

        # Adaptive residual mixing
        out = x + gate * mixed

        return out


# ============================================================
# 6. Adaptive Fusion
# ============================================================

class AdaptiveFusion(nn.Module):
    """
    Learns the contribution of:
        PatchMixer
        Linear Trend
    """

    def __init__(
        self,
        pred_len,
        hidden_dim=64
    ):
        super().__init__()

        self.gate = nn.Sequential(

            nn.Linear(
                pred_len * 2,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                pred_len
            ),

            nn.Sigmoid()
        )

    def forward(
        self,
        patch_forecast,
        trend_forecast
    ):
        """
        Both:
            [B, N, H]
        """

        combined = torch.cat(
            [
                patch_forecast,
                trend_forecast
            ],
            dim=-1
        )

        # Horizon-specific gate
        alpha = self.gate(
            combined
        )

        # Horizon-aware fusion
        output = (
            alpha * patch_forecast
            +
            (1.0 - alpha)
            * trend_forecast
        )

        return output, alpha


# ============================================================
# 7. Main Backbone
# ============================================================

class Backbone(nn.Module):

    def __init__(
        self,
        configs,
        revin=True,
        affine=True,
        subtract_last=False
    ):
        super().__init__()

        self.nvars = configs.enc_in
        self.lookback = configs.seq_len
        self.forecasting = configs.pred_len

        self.d_model = getattr(
            configs,
            "d_model",
            128
        )

        self.depth = getattr(
            configs,
            "e_layers",
            2
        )

        self.dropout_rate = getattr(
            configs,
            "dropout",
            0.1
        )

        self.head_dropout = getattr(
            configs,
            "head_dropout",
            0.0
        )

        self.kernel_size = getattr(
            configs,
            "mixer_kernel_size",
            8
        )

        # ----------------------------------------------------
        # Multi-scale settings
        # ----------------------------------------------------

        self.patch_sizes = getattr(
            configs,
            "patch_sizes",
            [8, 16, 32]
        )

        self.strides = getattr(
            configs,
            "patch_strides",
            [4, 8, 16]
        )

        # ----------------------------------------------------
        # Decomposition
        # ----------------------------------------------------

        self.trend_kernel = getattr(
            configs,
            "trend_kernel",
            25
        )

        self.decomposition = (
            SeriesDecomposition(
                self.trend_kernel
            )
        )

        # ----------------------------------------------------
        # Multi-scale PatchMixer
        # ----------------------------------------------------

        self.patch_mixer = MultiScalePatchMixer(

            seq_len=self.lookback,

            pred_len=self.forecasting,

            d_model=self.d_model,

            depth=self.depth,

            dropout=self.dropout_rate,

            head_dropout=self.head_dropout,

            kernel_size=self.kernel_size,

            patch_sizes=self.patch_sizes,

            strides=self.strides
        )

        # ----------------------------------------------------
        # Trend branch
        # ----------------------------------------------------

        self.trend_branch = LinearTrendBranch(

            seq_len=self.lookback,

            pred_len=self.forecasting,

            nvars=self.nvars,

            dropout=self.dropout_rate
        )

        # ----------------------------------------------------
        # Adaptive fusion
        # ----------------------------------------------------

        self.adaptive_fusion = AdaptiveFusion(

            pred_len=self.forecasting,

            hidden_dim=getattr(
                configs,
                "fusion_hidden",
                64
            )
        )

        # ----------------------------------------------------
        # Adaptive channel mixing
        # ----------------------------------------------------

        self.channel_gate = AdaptiveChannelGate(

            nvars=self.nvars,

            pred_len=self.forecasting,

            hidden_dim=getattr(
                configs,
                "channel_hidden",
                64
            ),

            dropout=self.dropout_rate
        )

        # ----------------------------------------------------
        # Final projection
        # ----------------------------------------------------

        self.output_projection = nn.Sequential(

            nn.Linear(
                self.forecasting,
                self.forecasting
            ),

            nn.GELU(),

            nn.Dropout(
                self.head_dropout
            )
        )

        # ----------------------------------------------------
        # RevIN
        # ----------------------------------------------------

        self.revin = revin

        if self.revin:

            self.revin_layer = RevIN(
                self.nvars,
                affine=affine,
                subtract_last=subtract_last
            )

    def forward(
        self,
        x,
        return_gate=False
    ):
        """
        Input:
            x [B, L, N]

        Output:
            [B, H, N]
        """

        # ----------------------------------------------------
        # RevIN normalization
        # ----------------------------------------------------

        if self.revin:
            x = self.revin_layer(
                x,
                'norm'
            )

        # ----------------------------------------------------
        # Decomposition
        # ----------------------------------------------------

        trend, residual = self.decomposition(x)

        # ----------------------------------------------------
        # PatchMixer operates on residual
        # ----------------------------------------------------

        residual = residual.permute(
            0, 2, 1
        )

        patch_forecast = self.patch_mixer(
            residual
        )

        # ----------------------------------------------------
        # Trend branch
        # ----------------------------------------------------

        trend_forecast = self.trend_branch(
            trend
        )

        # ----------------------------------------------------
        # Adaptive fusion
        # ----------------------------------------------------

        fused, alpha = self.adaptive_fusion(

            patch_forecast,

            trend_forecast
        )

        # ----------------------------------------------------
        # Adaptive channel mixing
        # ----------------------------------------------------

        fused = self.channel_gate(
            fused
        )

        # ----------------------------------------------------
        # Final projection
        # ----------------------------------------------------

        fused = self.output_projection(
            fused
        )

        # ----------------------------------------------------
        # [B, N, H] -> [B, H, N]
        # ----------------------------------------------------

        output = fused.permute(
            0, 2, 1
        )

        # ----------------------------------------------------
        # RevIN denormalization
        # ----------------------------------------------------

        if self.revin:

            output = self.revin_layer(
                output,
                'denorm'
            )

        if return_gate:
            return output, alpha

        return output


# ============================================================
# 8. Model Wrapper
# ============================================================

class Model(nn.Module):

    def __init__(self, configs):

        super().__init__()

        self.model = Backbone(
            configs
        )

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        mask=None
    ):

        return self.model(
            x_enc
        )
