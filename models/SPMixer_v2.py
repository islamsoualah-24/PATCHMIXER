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
# Statistical Profiling Module  (unchanged from v1)
# =============================================================================
class StatisticalProfiler(nn.Module):
    """
    Computes four global statistical descriptors per sample from the raw
    (pre-RevIN) input window.  All operations are differentiable.

    Input : x     [B, L, N]  — batch, seq_len, n_vars
    Output: stats [B, 4]     — (Trend Strength, Seasonality Strength,
                                 Distribution Shift, Multivariate Correlation Index)
    """

    def __init__(self, period: int = 24, eps: float = 1e-6):
        super().__init__()
        self.period = period
        self.eps = eps

    @staticmethod
    def _moving_average(x: Tensor, window: int) -> Tensor:
        """Symmetric 1-D average pooling along the time axis."""
        x_ = x.permute(0, 2, 1)                           # [B, N, L]
        pad_l = window // 2
        pad_r = window - 1 - pad_l
        x_pad = F.pad(x_, (pad_l, pad_r), mode='replicate')
        trend = F.avg_pool1d(x_pad, kernel_size=window, stride=1)
        return trend.permute(0, 2, 1)                      # [B, L, N]

    def _trend_strength(self, x: Tensor) -> Tensor:
        """FT = 1 - Var(residual) / Var(x),  clipped to [0, 1]."""
        window = max(3, min(self.period, x.shape[1] // 2 if x.shape[1] > 4 else 3))
        trend  = self._moving_average(x, window)
        resid  = x - trend
        var_r  = resid.var(dim=1, unbiased=False)
        var_x  = (resid + trend - trend.mean(dim=1, keepdim=True)).var(dim=1, unbiased=False)
        ft     = (1.0 - var_r / (var_x + self.eps)).clamp(0.0, 1.0).mean(dim=-1)  # [B]
        return ft

    def _seasonality_strength(self, x: Tensor) -> Tensor:
        """FS = dominant-bin power / total FFT power, averaged over channels."""
        x_c   = x - x.mean(dim=1, keepdim=True)
        spec  = torch.fft.rfft(x_c, dim=1)
        power = spec.real ** 2 + spec.imag ** 2            # [B, F, N]
        power = power[:, 1:, :] if power.shape[1] > 1 else power   # drop DC
        fs    = (power.max(dim=1).values /
                 (power.sum(dim=1) + self.eps)).clamp(0.0, 1.0).mean(dim=-1)  # [B]
        return fs

    def _distribution_shift(self, x: Tensor) -> Tensor:
        """Normalised Δmean + Δstd between the first and second halves."""
        L = x.shape[1]
        half   = max(1, L // 2)
        first, second = x[:, :half], x[:, half:]
        d_mean = (first.mean(dim=1) - second.mean(dim=1)).abs()
        d_std  = (first.std(dim=1, unbiased=False) -
                  second.std(dim=1, unbiased=False)).abs()
        scale  = x.std(dim=1, unbiased=False) + self.eps
        return torch.tanh(((d_mean + d_std) / scale).mean(dim=-1))   # [B] ∈ [0,1)

    def _multivariate_correlation_index(self, x: Tensor) -> Tensor:
        """Mean absolute off-diagonal Pearson correlation across channels."""
        B, L, N = x.shape
        if N == 1:
            return torch.zeros(B, device=x.device, dtype=x.dtype)
        x_c  = x - x.mean(dim=1, keepdim=True)
        std  = x_c.std(dim=1, unbiased=False) + self.eps
        x_n  = x_c / std.unsqueeze(1)
        corr = torch.einsum('bln,blm->bnm', x_n, x_n) / L             # [B, N, N]
        eye  = torch.eye(N, device=x.device, dtype=torch.bool)
        off  = corr.masked_select(
            ~eye.unsqueeze(0).expand(B, -1, -1)
        ).view(B, -1)
        return off.abs().mean(dim=-1).clamp(0.0, 1.0)                  # [B]

    def forward(self, x: Tensor) -> Tensor:
        ft    = self._trend_strength(x)
        fs    = self._seasonality_strength(x)
        shift = self._distribution_shift(x)
        mci   = self._multivariate_correlation_index(x)
        return torch.stack([ft, fs, shift, mci], dim=-1)               # [B, 4]


# =============================================================================
# Dynamic Statistical Router
# =============================================================================
class DynamicStatisticalRouter(nn.Module):
    """
    Encodes the 4-dim statistical vector and produces four families of signals
    that drive every downstream conditioning step:

        h               [B, hidden_dim]            shared stat encoding
        routing_weights [B, 3]                     Softmax branch weights (Σ=1)
        feature_alpha   [B, 3, d_model]            per-branch per-feature scales ∈(0,1)
        gammas          [B, n_film_levels, d_model] FiLM scale (centred at 1 at init)
        betas           [B, n_film_levels, d_model] FiLM shift (centred at 0 at init)

    FiLM injection levels  (n_film_levels = depth + 2):
        level 0           : right after W_P   (pre-backbone)
        levels 1 .. depth : before each PatchMixerLayer
        level depth+1     : before head1      (pre-head)
    """

    def __init__(self, stat_dim: int, d_model: int, depth: int, hidden_dim: int = 32):
        super().__init__()
        self.d_model   = d_model
        self.n_levels  = depth + 2        # pre-backbone + per-block + pre-head

        # ── Shared 2-layer encoder ────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # ── (a) Softmax routing: 3 expert branches ────────────────────────────
        self.routing_head = nn.Linear(hidden_dim, 3)

        # ── (b) Multi-dim residual alpha: per-branch per-feature ──────────────
        self.alpha_head = nn.Linear(hidden_dim, 3 * d_model)

        # ── (c) FiLM params for all levels packed in one projection ───────────
        # output: [B, n_levels * 2 * d_model]  →  reshape  →  gammas, betas
        self.film_head = nn.Linear(hidden_dim, self.n_levels * 2 * d_model)

        # Near-identity FiLM initialisation (gamma≈1, beta≈0)
        nn.init.normal_(self.film_head.weight, std=0.01)
        nn.init.zeros_(self.film_head.bias)

    def forward(self, stats: Tensor):
        B = stats.shape[0]
        h = self.encoder(stats)                                         # [B, H]

        # (a) Softmax routing weights — sum to 1 over the 3 expert branches
        routing_weights = F.softmax(self.routing_head(h), dim=-1)       # [B, 3]

        # (b) Per-branch per-feature scale factors ∈ (0, 1)  via sigmoid
        feature_alpha = torch.sigmoid(
            self.alpha_head(h)
        ).view(B, 3, self.d_model)                                      # [B, 3, D]

        # (c) FiLM: gamma centred at 1, beta centred at 0 (near-identity at init)
        film_raw = self.film_head(h).view(B, self.n_levels, 2, self.d_model)
        gammas = 1.0 + film_raw[:, :, 0, :]                            # [B, L, D]
        betas  =       film_raw[:, :, 1, :]                            # [B, L, D]

        return h, routing_weights, feature_alpha, gammas, betas


# =============================================================================
# Expert Branches
# =============================================================================
class ExpertBranches(nn.Module):
    """
    Three parallel lightweight expert branches.
    All branches:  input  [B*N, P, d_model]  →  output [B*N, P, d_model]

    Branch 1 — Linear           : pointwise feature projection
    Branch 2 — DS-CNN           : depthwise-separable convolution along patches
    Branch 3 — Lightweight Attn : multi-head self-attention at reduced inner dim
    """

    def __init__(self, d_model: int, cnn_kernel: int = 3, attn_heads: int = 2):
        super().__init__()

        # inner attention dim: small, divisible by attn_heads
        attn_dim = (min(d_model, 64) // attn_heads) * attn_heads
        self.attn_heads = attn_heads
        self.attn_dim   = attn_dim
        self.attn_scale = attn_dim ** -0.5

        # ── Branch 1: Linear ─────────────────────────────────────────────────
        self.lin_proj = nn.Linear(d_model, d_model)
        self.lin_norm = nn.LayerNorm(d_model)

        # ── Branch 2: Depthwise-Separable CNN ────────────────────────────────
        # d_model → channels, patch_num → spatial length
        self.dw_conv  = nn.Conv1d(d_model, d_model, kernel_size=cnn_kernel,
                                   padding='same', groups=d_model, bias=False)
        self.pw_conv  = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.cnn_norm = nn.LayerNorm(d_model)

        # ── Branch 3: Lightweight Self-Attention ─────────────────────────────
        # QKV projected to a smaller attn_dim to reduce compute
        self.attn_q    = nn.Linear(d_model, attn_dim, bias=False)
        self.attn_k    = nn.Linear(d_model, attn_dim, bias=False)
        self.attn_v    = nn.Linear(d_model, attn_dim, bias=False)
        self.attn_out  = nn.Linear(attn_dim, d_model)
        self.attn_norm = nn.LayerNorm(d_model)

    def _self_attention(self, x: Tensor) -> Tensor:
        """Multi-head dot-product self-attention at reduced dimensionality."""
        BN, P, _ = x.shape
        H, dh = self.attn_heads, self.attn_dim // self.attn_heads

        # Project and split into heads
        Q = self.attn_q(x).view(BN, P, H, dh).transpose(1, 2)   # [BN, H, P, dh]
        K = self.attn_k(x).view(BN, P, H, dh).transpose(1, 2)
        V = self.attn_v(x).view(BN, P, H, dh).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) * self.attn_scale      # [BN, H, P, P]
        out = F.softmax(scores, dim=-1) @ V                        # [BN, H, P, dh]
        out = out.transpose(1, 2).reshape(BN, P, -1)               # [BN, P, attn_dim]
        return self.attn_out(out)                                   # [BN, P, D]

    def forward(self, x: Tensor):
        # ── Branch 1: Linear ─────────────────────────────────────────────────
        b1 = self.lin_norm(F.gelu(self.lin_proj(x)))               # [BN, P, D]

        # ── Branch 2: Depthwise-Separable CNN ────────────────────────────────
        xc = x.permute(0, 2, 1)                                    # [BN, D, P]
        xc = F.gelu(self.pw_conv(self.dw_conv(xc)))
        b2 = self.cnn_norm(xc.permute(0, 2, 1))                   # [BN, P, D]

        # ── Branch 3: Lightweight Self-Attention ─────────────────────────────
        b3 = self.attn_norm(self._self_attention(x))               # [BN, P, D]

        return b1, b2, b3


# =============================================================================
# Statistical Cross-Attention
# =============================================================================
class StatisticalCrossAttention(nn.Module):
    """
    Temporal patch embeddings act as Queries; statistical descriptor tokens
    act as Keys and Values — giving each patch position the ability to
    selectively pull global statistical context.

    The 4 statistical scalars are jointly projected into a bank of
    4 tokens of size d_model.  A learnable gate (init=0) ensures this
    module starts as an identity and is switched on gradually by the
    optimiser.

    Input:
        x     [B*N, P, d_model]
        stats [B,   4]
    Output:
        x     [B*N, P, d_model]
    """

    def __init__(self, d_model: int, stat_dim: int = 4, attn_heads: int = 2):
        super().__init__()
        attn_dim = (min(d_model, 64) // attn_heads) * attn_heads
        self.attn_heads = attn_heads
        self.attn_dim   = attn_dim
        self.attn_scale = attn_dim ** -0.5
        self.stat_dim   = stat_dim
        self.d_model    = d_model

        # Project stat vector [B, 4] → token bank [B, 4, d_model]
        self.stat_token_proj = nn.Linear(stat_dim, stat_dim * d_model)

        # Q from temporal, K / V from stat tokens — all projected to attn_dim
        self.q_proj   = nn.Linear(d_model, attn_dim, bias=False)
        self.k_proj   = nn.Linear(d_model, attn_dim, bias=False)
        self.v_proj   = nn.Linear(d_model, attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, d_model)

        self.norm = nn.LayerNorm(d_model)
        # Scalar gate init=0 → tanh(0)=0 → no effect at initialisation
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, stats: Tensor) -> Tensor:
        BN, P, D = x.shape
        B  = stats.shape[0]
        N  = BN // B
        H  = self.attn_heads
        dh = self.attn_dim // H

        # ── Build stat token bank [B*N, 4, D] ───────────────────────────────
        stat_tokens = self.stat_token_proj(stats).view(B, self.stat_dim, D)
        stat_tokens = stat_tokens.repeat_interleave(N, dim=0)          # [B*N, 4, D]

        # ── Cross-Attention: Q=temporal, K/V=stat tokens ─────────────────────
        Q = self.q_proj(x).view(BN, P, H, dh).transpose(1, 2)         # [BN, H, P,  dh]
        K = self.k_proj(stat_tokens).view(BN, 4, H, dh).transpose(1, 2) # [BN, H, 4, dh]
        V = self.v_proj(stat_tokens).view(BN, 4, H, dh).transpose(1, 2)

        attn = F.softmax((Q @ K.transpose(-2, -1)) * self.attn_scale,
                         dim=-1)                                        # [BN, H, P, 4]
        out  = (attn @ V).transpose(1, 2).reshape(BN, P, -1)           # [BN, P, attn_dim]
        out  = self.out_proj(out)                                        # [BN, P, D]

        # ── Gated residual (gate ≈ 0 at init → no initial distortion) ────────
        return self.norm(x + torch.tanh(self.gate) * out)


# =============================================================================
# Statistical Embedding
# =============================================================================
class StatisticalEmbedding(nn.Module):
    """Projects raw statistical descriptors [B, 4] → [B, embed_dim]."""

    def __init__(self, stat_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(stat_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, stats: Tensor) -> Tensor:
        return self.proj(stats)                                         # [B, embed_dim]


# =============================================================================
# Adaptive Cross Fusion  (replaces the old AdaptiveFusion)
# =============================================================================
class AdaptiveFusion(nn.Module):
    """
    Attention-based fusion of the PatchMixer temporal forecast vector with the
    statistical embedding.

    temporal_feat → Query
    stat_embed    → Key & Value

    A dot-product attention score gates the value contribution; a learnable
    gating MLP further controls which forecast dimensions receive statistical
    modulation.  A residual connection ensures the temporal pathway is
    preserved.

    Input:
        temporal_feat [B*N, forecast_dim]
        stat_embed    [B,   stat_embed_dim]
    Output:
                      [B*N, forecast_dim]
    """

    def __init__(self, forecast_dim: int, stat_embed_dim: int, attn_dim: int = 32):
        super().__init__()
        self.scale = attn_dim ** -0.5

        # Attention projections
        self.q_proj = nn.Linear(forecast_dim,   attn_dim)
        self.k_proj = nn.Linear(stat_embed_dim, attn_dim)
        self.v_proj = nn.Linear(stat_embed_dim, forecast_dim)

        # Per-dimension gating: decides which forecast dims absorb stat signal
        self.gate_proj = nn.Sequential(
            nn.Linear(forecast_dim + stat_embed_dim, forecast_dim),
            nn.Sigmoid(),
        )

        self.norm  = nn.LayerNorm(forecast_dim)
        # Global mixing coeff init=0 → sigmoid(0)=0.5, but multiplied by near-zero
        # stat_contribution early in training → safe warm-up
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, temporal_feat: Tensor, stat_embed: Tensor) -> Tensor:
        BN = temporal_feat.shape[0]
        B  = stat_embed.shape[0]
        N  = BN // B
        se = stat_embed.repeat_interleave(N, dim=0)                     # [B*N, E]

        # Dot-product attention score (scalar per sample)
        Q      = self.q_proj(temporal_feat)                             # [B*N, attn_dim]
        K      = self.k_proj(se)                                        # [B*N, attn_dim]
        attn_w = torch.sigmoid((Q * K).sum(-1, keepdim=True) * self.scale)  # [B*N, 1]

        # Value: stat embedding projected to forecast space
        V    = self.v_proj(se)                                          # [B*N, F]
        gate = self.gate_proj(torch.cat([temporal_feat, se], dim=-1))  # [B*N, F]

        # Stat contribution modulated by attention score and per-dim gate
        stat_contribution = attn_w * gate * V                           # [B*N, F]

        # Gated residual add — alpha starts near 0, grows during training
        return self.norm(
            temporal_feat + torch.sigmoid(self.alpha) * stat_contribution
        )


# =============================================================================
# FiLM conditioning helper
# =============================================================================
def _apply_film(x: Tensor, gamma: Tensor, beta: Tensor, n_vars: int) -> Tensor:
    """
    Feature-wise Linear Modulation:  x = gamma * x + beta

    x:     [B*N, T, d_model]
    gamma: [B,   d_model]     ← router output for a specific level
    beta:  [B,   d_model]
    """
    g = gamma.repeat_interleave(n_vars, dim=0).unsqueeze(1)   # [B*N, 1, D]
    b = beta .repeat_interleave(n_vars, dim=0).unsqueeze(1)
    return g * x + b


# =============================================================================
# PatchMixerLayer  — UNCHANGED
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
        x = x + self.Resnet(x)    # [B*N, patch_num, d_model]
        x = self.Conv_1x1(x)      # [B*N, a,         d_model]
        return x


# =============================================================================
# Model wrapper  — UNCHANGED
# =============================================================================
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = Backbone(configs)

    def forward(self, x):
        return self.model(x)


# =============================================================================
# Backbone
# =============================================================================
class Backbone(nn.Module):
    """
    PatchMixer backbone with an optional Dynamic Statistical Routing system
    controlled by configs.use_stat_router (default False).

    When enabled the forward pass executes this pipeline:

        Raw x  →  StatisticalProfiler  →  DynamicStatisticalRouter
                                                │
               ┌───────────────────────────────┤
               │ FiLM level-0 (pre-backbone)   │
               ↓                               │
        W_P embeddings  →  ExpertBranches      │  routing_weights, feature_alpha
                               │               │  gammas, betas
               ┌───────────────┤               │
               │  Residual expert mixing        │
               ↓                               │
        Shortcut  →  head0 ────────────────────┤
               │                               │
               ├─ FiLM + PatchMixerBlock × d ──┤
               │                               │
               ↓                               │
        StatisticalCrossAttention              │
               │                               │
               ├─ FiLM level depth+1           │
               ↓                               │
        head1 + shortcut  →  AdaptiveFusion   ←┘ (stat_embedding)
               │
               ↓
           Forecast

    All original backbone attributes, heads, RevIN and training pipeline
    are completely unchanged.
    """

    def __init__(self, configs, revin=True, affine=True, subtract_last=False):
        super().__init__()

        self.nvals        = configs.enc_in
        self.lookback     = configs.seq_len
        self.forecasting  = configs.pred_len
        self.patch_size   = configs.patch_len
        self.stride       = configs.stride
        self.kernel_size  = configs.mixer_kernel_size

        self.PatchMixer_blocks   = nn.ModuleList([])
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num = int((self.lookback - self.patch_size) / self.stride + 1) + 1
        self.a         = self.patch_num
        self.d_model     = configs.d_model
        self.dropout     = configs.dropout
        self.head_dropout = configs.head_dropout
        self.depth       = configs.e_layers

        for _ in range(self.depth):
            self.PatchMixer_blocks.append(
                PatchMixerLayer(dim=self.patch_num, a=self.a,
                                kernel_size=self.kernel_size)
            )

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
        # NOTE: self.dropout is deliberately reused: first holds a float, then
        # an nn.Dropout module — preserving the original code pattern exactly.
        self.dropout = nn.Dropout(self.dropout)

        # RevIN
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(self.nvals, affine=affine,
                                     subtract_last=subtract_last)

        # -----------------------------------------------------------------------
        # Dynamic Statistical Routing (optional — enable via configs.use_stat_router)
        # -----------------------------------------------------------------------
        self.use_stat_router = getattr(configs, 'use_stat_router', False)

        if self.use_stat_router:
            # Config knobs (all have safe defaults)
            stat_period     = getattr(configs, 'stat_period',          24)
            stat_embed_dim  = getattr(configs, 'stat_embed_dim',       32)
            router_hidden   = getattr(configs, 'stat_router_hidden',   32)
            cnn_kernel      = getattr(configs, 'stat_cnn_kernel',       3)
            fusion_attn_dim = getattr(configs, 'stat_fusion_attn_dim', 32)

            self.stat_profiler   = StatisticalProfiler(period=stat_period)

            self.stat_router     = DynamicStatisticalRouter(
                stat_dim=4,
                d_model=self.d_model,
                depth=self.depth,
                hidden_dim=router_hidden,
            )
            self.expert_branches = ExpertBranches(
                d_model=self.d_model,
                cnn_kernel=cnn_kernel,
            )
            self.stat_cross_attn = StatisticalCrossAttention(
                d_model=self.d_model,
                stat_dim=4,
            )
            self.stat_embedding  = StatisticalEmbedding(
                stat_dim=4,
                embed_dim=stat_embed_dim,
            )
            self.adaptive_fusion = AdaptiveFusion(
                forecast_dim=self.forecasting,
                stat_embed_dim=stat_embed_dim,
                attn_dim=fusion_attn_dim,
            )

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        bs    = x.shape[0]
        nvars = x.shape[-1]

        # ── 1. Statistical Profiling (raw window, before RevIN) ───────────────
        stats = None
        if self.use_stat_router:
            stats = self.stat_profiler(x)                               # [B, 4]
            _, routing_weights, feature_alpha, gammas, betas = \
                self.stat_router(stats)

            # Expand per-variable for the B*N dimension used inside the backbone
            # (done once here to avoid repeated interleave calls later)
            rw = routing_weights.repeat_interleave(nvars, dim=0)        # [B*N, 3]
            fa = feature_alpha.repeat_interleave(nvars, dim=0)          # [B*N, 3, D]

        # ── 2. RevIN normalisation ─────────────────────────────────────────────
        if self.revin:
            x = self.revin_layer(x, 'norm')
        x = x.permute(0, 2, 1)                                          # [B, N, L]

        # ── 3. Patch extraction & embedding ───────────────────────────────────
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # x: [B, N, patch_num, patch_size]
        x = self.W_P(x)                                                  # [B, N, P, D]
        x = torch.reshape(x, (bs * nvars, x.shape[2], x.shape[3]))      # [B*N, P, D]
        x = self.dropout(x)

        if self.use_stat_router:
            # ── 4. FiLM level 0: pre-backbone conditioning ────────────────────
            # gamma, beta: [B, D]  →  broadcast to [B*N, 1, D]
            x = _apply_film(x, gammas[:, 0], betas[:, 0], nvars)        # [B*N, P, D]

            # ── 5. Multi-branch routing + residual expert mixing ───────────────
            b1, b2, b3 = self.expert_branches(x)                        # each [B*N, P, D]

            # alpha_i = routing_weight_i [B*N, 1] × feature_alpha_i [B*N, D]
            # → [B*N, D]  →  unsqueeze  →  [B*N, 1, D]  (broadcasts over P)
            a1 = (rw[:, 0:1] * fa[:, 0]).unsqueeze(1)                   # [B*N, 1, D]
            a2 = (rw[:, 1:2] * fa[:, 1]).unsqueeze(1)
            a3 = (rw[:, 2:3] * fa[:, 2]).unsqueeze(1)

            # Residual expert mixing: x = x + Σ alpha_i · branch_i
            x = x + a1 * b1 + a2 * b2 + a3 * b3                        # [B*N, P, D]

        # ── 6. Shortcut (linear probe on pre-backbone representation) ──────────
        u = self.head0(x)                                                # [B*N, pred_len]

        # ── 7. PatchMixer blocks with per-block FiLM conditioning ─────────────
        for i, block in enumerate(self.PatchMixer_blocks):
            if self.use_stat_router:
                # FiLM level i+1  (levels 1..depth)
                x = _apply_film(x, gammas[:, i + 1], betas[:, i + 1], nvars)
            x = block(x)                                                 # [B*N, P, D]

        # ── 8. Statistical Cross-Attention (temporal Q, stat K/V) ─────────────
        if self.use_stat_router:
            x = self.stat_cross_attn(x, stats)                          # [B*N, P, D]

            # ── 9. FiLM level depth+1: pre-head conditioning ──────────────────
            x = _apply_film(x,
                            gammas[:, self.depth + 1],
                            betas[:, self.depth + 1],
                            nvars)

        # ── 10. Prediction heads ──────────────────────────────────────────────
        x = self.head1(x)                                                # [B*N, pred_len]
        x = u + x                                                        # residual shortcut

        # ── 11. Adaptive Cross-Fusion: fuse temporal + statistical embedding ───
        if self.use_stat_router:
            stat_embed = self.stat_embedding(stats)                      # [B, embed_dim]
            x = self.adaptive_fusion(x, stat_embed)                     # [B*N, pred_len]

        # ── 12. Reshape & RevIN denormalisation ───────────────────────────────
        x = torch.reshape(x, (bs, nvars, -1))                           # [B, N, pred_len]
        x = x.permute(0, 2, 1)                                          # [B, pred_len, N]
        if self.revin:
            x = self.revin_layer(x, 'denorm')
        return x
