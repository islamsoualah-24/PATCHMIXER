__all__ = ['PatchMixer']

# Cell
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from layers.PatchTST_layers import *
from layers.RevIN import RevIN


# =============================================================================
# Statistical Profiling Module
# =============================================================================
class StatisticalProfiler(nn.Module):
    """
    Computes global statistical descriptors from the raw input window, per
    sample in the batch. All operations are differentiable (built from
    mean / var / fft / corrcoef-style primitives) so the resulting vector
    can be backpropagated through if needed, even though in practice it is
    treated as a conditioning signal.

    Input:  x [B, L, N]   (batch, seq_len, n_vars) -- pre-RevIN raw window
    Output: stats [B, 4]  -> (Trend Strength, Seasonality Strength,
                               Distribution Shift, Multivariate Correlation Index)
    """
    def __init__(self, period: int = 24, eps: float = 1e-6):
        super().__init__()
        self.period = period
        self.eps = eps

    @staticmethod
    def _moving_average(x: Tensor, window: int) -> Tensor:
        # x: [B, L, N] -> trend via 1D average pooling along time, same length
        B, L, N = x.shape
        x_ = x.permute(0, 2, 1)                      # [B, N, L]
        pad_l = window // 2
        pad_r = window - 1 - pad_l
        x_pad = F.pad(x_, (pad_l, pad_r), mode='replicate')
        trend = F.avg_pool1d(x_pad, kernel_size=window, stride=1)  # [B, N, L]
        return trend.permute(0, 2, 1)                 # [B, L, N]

    def _trend_strength(self, x: Tensor) -> Tensor:
        # FT = 1 - Var(residual) / Var(detrended_input), clipped to [0, 1]
        window = max(3, min(self.period, x.shape[1] // 2 if x.shape[1] > 4 else 3))
        trend = self._moving_average(x, window)
        resid = x - trend
        var_resid = resid.var(dim=1, unbiased=False)                  # [B, N]
        var_detrended = (resid + trend - trend.mean(dim=1, keepdim=True)).var(dim=1, unbiased=False)
        ft = 1.0 - var_resid / (var_detrended + self.eps)
        ft = ft.clamp(0.0, 1.0).mean(dim=-1)                           # [B]
        return ft

    def _seasonality_strength(self, x: Tensor) -> Tensor:
        # FS via dominant-frequency power ratio of the FFT spectrum
        B, L, N = x.shape
        x_centered = x - x.mean(dim=1, keepdim=True)
        spec = torch.fft.rfft(x_centered, dim=1)
        power = (spec.real ** 2 + spec.imag ** 2)                      # [B, F, N]
        # ignore DC component
        power = power[:, 1:, :] if power.shape[1] > 1 else power
        total_power = power.sum(dim=1) + self.eps                      # [B, N]
        dominant_power = power.max(dim=1).values                       # [B, N]
        fs = (dominant_power / total_power).clamp(0.0, 1.0).mean(dim=-1)  # [B]
        return fs

    def _distribution_shift(self, x: Tensor) -> Tensor:
        # compares first-half vs second-half mean/std (normalized distance)
        L = x.shape[1]
        half = max(1, L // 2)
        first, second = x[:, :half, :], x[:, half:, :]
        mean_diff = (first.mean(dim=1) - second.mean(dim=1)).abs()
        std_diff = (first.std(dim=1, unbiased=False) - second.std(dim=1, unbiased=False)).abs()
        scale = x.std(dim=1, unbiased=False) + self.eps
        shift = ((mean_diff + std_diff) / scale).mean(dim=-1)          # [B]
        return torch.tanh(shift)                                        # squashed to [0,1)

    def _multivariate_correlation_index(self, x: Tensor) -> Tensor:
        # mean absolute off-diagonal Pearson correlation across channels
        B, L, N = x.shape
        if N == 1:
            return torch.zeros(B, device=x.device, dtype=x.dtype)
        x_centered = x - x.mean(dim=1, keepdim=True)
        std = x_centered.std(dim=1, unbiased=False) + self.eps         # [B, N]
        x_norm = x_centered / std.unsqueeze(1)                         # [B, L, N]
        corr = torch.einsum('bln,blm->bnm', x_norm, x_norm) / L        # [B, N, N]
        eye = torch.eye(N, device=x.device, dtype=torch.bool)
        off_diag = corr.masked_select(~eye.unsqueeze(0).expand(B, -1, -1)).view(B, -1)
        mci = off_diag.abs().mean(dim=-1).clamp(0.0, 1.0)               # [B]
        return mci

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, L, N] raw input window (before RevIN normalization)
        ft = self._trend_strength(x)
        fs = self._seasonality_strength(x)
        shift = self._distribution_shift(x)
        mci = self._multivariate_correlation_index(x)
        stats = torch.stack([ft, fs, shift, mci], dim=-1)              # [B, 4]
        return stats


# =============================================================================
# Statistical Router
# =============================================================================
class StatisticalRouter(nn.Module):
    """
    Consumes the 4-dim statistical descriptor vector and produces lightweight,
    sample-adaptive routing signals that modulate the PatchMixer's internal
    feature representations:
      - feature-wise scaling gate over d_model       (channel-wise gating)
      - patch-wise attention-style modulation weight (attention modulation)
      - scalar residual mixing coefficient            (adaptive residual weighting)

    The router is intentionally tiny (a 2-layer MLP) to respect the
    "lightweight module" constraint.
    """
    def __init__(self, stat_dim: int, d_model: int, patch_num: int, hidden_dim: int = 16):
        super().__init__()
        self.d_model = d_model
        self.patch_num = patch_num

        self.encoder = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.GELU(),
        )
        self.feature_gate = nn.Linear(hidden_dim, d_model)     # channel-wise gating (over d_model)
        self.patch_gate = nn.Linear(hidden_dim, patch_num)     # attention-style modulation (over patches)
        self.residual_alpha = nn.Linear(hidden_dim, 1)         # adaptive residual weighting scalar

    def forward(self, stats: Tensor):
        """
        stats: [B, stat_dim]
        returns:
            feat_gate: [B, 1, d_model]   in (0, 2)  -- centered around 1
            patch_gate: [B, patch_num, 1] in (0, 2)  -- centered around 1
            alpha: [B, 1, 1] in (0, 1)    -- residual mixing coefficient
        """
        h = self.encoder(stats)                                     # [B, hidden]
        feat_gate = 2 * torch.sigmoid(self.feature_gate(h))         # [B, d_model], centered at 1
        patch_gate = 2 * torch.sigmoid(self.patch_gate(h))          # [B, patch_num], centered at 1
        alpha = torch.sigmoid(self.residual_alpha(h))                # [B, 1]

        feat_gate = feat_gate.unsqueeze(1)                           # [B, 1, d_model]
        patch_gate = patch_gate.unsqueeze(-1)                        # [B, patch_num, 1]
        alpha = alpha.unsqueeze(-1)                                  # [B, 1, 1]
        return feat_gate, patch_gate, alpha


# =============================================================================
# Statistical Embedding + Adaptive Fusion
# =============================================================================
class StatisticalEmbedding(nn.Module):
    """Projects the raw statistical descriptors into a learned embedding."""
    def __init__(self, stat_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(stat_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, stats: Tensor) -> Tensor:
        return self.proj(stats)   # [B, embed_dim]


class AdaptiveFusion(nn.Module):
    """
    Lightweight fusion of PatchMixer temporal features with the statistical
    embedding. A learnable gate (computed from both signals) determines how
    much the statistical pathway contributes to the final forecast, sample
    by sample.
    """
    def __init__(self, forecast_dim: int, stat_embed_dim: int):
        super().__init__()
        self.stat_to_forecast = nn.Linear(stat_embed_dim, forecast_dim)
        self.gate = nn.Sequential(
            nn.Linear(forecast_dim + stat_embed_dim, forecast_dim),
            nn.Sigmoid()
        )

    def forward(self, temporal_feat: Tensor, stat_embed: Tensor) -> Tensor:
        """
        temporal_feat: [B*N, forecast_dim]
        stat_embed:    [B, stat_embed_dim]  -> broadcast across N variables
        """
        BN = temporal_feat.shape[0]
        B = stat_embed.shape[0]
        n_vars = BN // B
        stat_embed_exp = stat_embed.repeat_interleave(n_vars, dim=0)   # [B*N, stat_embed_dim]

        stat_proj = self.stat_to_forecast(stat_embed_exp)               # [B*N, forecast_dim]
        g = self.gate(torch.cat([temporal_feat, stat_embed_exp], dim=-1))  # [B*N, forecast_dim]
        fused = temporal_feat + g * stat_proj
        return fused


# =============================================================================
# Original PatchMixer building blocks (kept intact)
# =============================================================================
class PatchMixerLayer(nn.Module):
    def __init__(self, dim, a, kernel_size=8):
        super().__init__()
        self.Resnet = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, padding='same'),
            nn.GELU(),
            nn.BatchNorm1d(dim)
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv1d(dim, a, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(a)
        )

    def forward(self, x):
        x = x + self.Resnet(x)                  # x: [batch * n_val, patch_num, d_model]
        x = self.Conv_1x1(x)                     # x: [batch * n_val, a, d_model]
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = Backbone(configs)

    def forward(self, x):
        x = self.model(x)
        return x


class Backbone(nn.Module):
    def __init__(self, configs, revin=True, affine=True, subtract_last=False):
        super().__init__()

        self.nvals = configs.enc_in
        self.lookback = configs.seq_len
        self.forecasting = configs.pred_len
        self.patch_size = configs.patch_len
        self.stride = configs.stride
        self.kernel_size = configs.mixer_kernel_size

        self.PatchMixer_blocks = nn.ModuleList([])
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num = int((self.lookback - self.patch_size) / self.stride + 1) + 1
        self.a = self.patch_num
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.head_dropout = configs.head_dropout
        self.depth = configs.e_layers
        for _ in range(self.depth):
            self.PatchMixer_blocks.append(PatchMixerLayer(dim=self.patch_num, a=self.a, kernel_size=self.kernel_size))
        self.W_P = nn.Linear(self.patch_size, self.d_model)
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
        self.dropout = nn.Dropout(self.dropout)

        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(self.nvals, affine=affine, subtract_last=subtract_last)

        # ---------------------------------------------------------------
        # Statistical Router / Adaptive Fusion (optional, config-gated)
        # ---------------------------------------------------------------
        self.use_stat_router = getattr(configs, 'use_stat_router', False)
        if self.use_stat_router:
            stat_period = getattr(configs, 'stat_period', 24)
            stat_embed_dim = getattr(configs, 'stat_embed_dim', 16)
            router_hidden = getattr(configs, 'stat_router_hidden', 16)

            self.stat_profiler = StatisticalProfiler(period=stat_period)
            self.stat_router = StatisticalRouter(
                stat_dim=4, d_model=self.d_model, patch_num=self.patch_num, hidden_dim=router_hidden
            )
            self.stat_embedding = StatisticalEmbedding(stat_dim=4, embed_dim=stat_embed_dim)
            self.adaptive_fusion = AdaptiveFusion(forecast_dim=self.forecasting, stat_embed_dim=stat_embed_dim)

    def forward(self, x):
        bs = x.shape[0]
        nvars = x.shape[-1]

        # ---- Statistical Profiling Module (uses raw, pre-RevIN window) ----
        stats = None
        if self.use_stat_router:
            stats = self.stat_profiler(x)                                   # [B, 4]

        if self.revin:
            x = self.revin_layer(x, 'norm')
        x = x.permute(0, 2, 1)                                               # x: [batch, n_val, seq_len]

        x_lookback = self.padding_patch_layer(x)
        x = x_lookback.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # x: [batch, n_val, patch_num, patch_size]

        x = self.W_P(x)                                                      # x: [batch, n_val, patch_num, d_model]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # x: [batch * n_val, patch_num, d_model]
        x = self.dropout(x)

        # ---- Statistical Router: modulate features before the backbone ----
        if self.use_stat_router:
            feat_gate, patch_gate, alpha = self.stat_router(stats)          # broadcast per-variable below
            feat_gate = feat_gate.repeat_interleave(nvars, dim=0)           # [B*N, 1, d_model]
            patch_gate = patch_gate.repeat_interleave(nvars, dim=0)         # [B*N, patch_num, 1]
            alpha = alpha.repeat_interleave(nvars, dim=0)                   # [B*N, 1, 1]

            x_gated = x * feat_gate * patch_gate
            # adaptive residual weighting: blend original vs. gated representation
            x = alpha * x_gated + (1 - alpha) * x

        u = self.head0(x)

        for PatchMixer_block in self.PatchMixer_blocks:
            x = PatchMixer_block(x)
        x = self.head1(x)
        x = u + x                                                           # x: [batch * n_val, forecasting]

        # ---- Adaptive Fusion: merge temporal features with statistical embedding ----
        if self.use_stat_router:
            stat_embed = self.stat_embedding(stats)                         # [B, stat_embed_dim]
            x = self.adaptive_fusion(x, stat_embed)                         # [batch * n_val, forecasting]

        x = torch.reshape(x, (bs, nvars, -1))                                # x: [batch, n_val, pred_len]
        x = x.permute(0, 2, 1)
        if self.revin:
            x = self.revin_layer(x, 'denorm')
        return x
