import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------------------------------------------------------------------
# Original DLinear building blocks (unchanged)
# ----------------------------------------------------------------------------
class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series"""
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    """Series decomposition block"""
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# ----------------------------------------------------------------------------
# Lightweight conditional denoiser
# ----------------------------------------------------------------------------
def timestep_embedding(t, dim):
    """Sinusoidal embedding of integer diffusion steps. t: [N] -> [N, dim]"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class FiLMBlock(nn.Module):
    """Residual MLP block modulated (FiLM) by timestep + history condition."""
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.film = nn.Linear(d, 2 * d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, h, c):
        scale, shift = self.film(c).chunk(2, dim=-1)
        return h + self.mlp(self.norm(h) * (1 + scale) + shift)


class ResidualDenoiser(nn.Module):
    """
    eps_theta(r_t, t | history, base forecast)
    Works on flattened series [N = B*C, pred_len] and is shared across channels,
    so its size does not grow with the number of variables.
    """
    def __init__(self, seq_len, pred_len, channels, d=128, layers=2):
        super().__init__()
        self.d = d
        self.in_proj = nn.Linear(pred_len, d)
        self.cond_enc = nn.Sequential(
            nn.Linear(seq_len + pred_len, d), nn.GELU(), nn.Linear(d, d))
        self.chan_emb = nn.Embedding(channels, d)
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([FiLMBlock(d) for _ in range(layers)])
        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, pred_len)
        nn.init.zeros_(self.out_proj.weight)  # start by predicting zero noise
        nn.init.zeros_(self.out_proj.bias)

    def encode_condition(self, cond, chan_idx):
        """History-derived condition embedding (computed once per sampling)."""
        return self.cond_enc(cond) + self.chan_emb(chan_idx)

    def forward(self, r_t, t, cond_emb):
        c = cond_emb + self.t_mlp(timestep_embedding(t, self.d))
        h = self.in_proj(r_t)
        for blk in self.blocks:
            h = blk(h, c)
        return self.out_proj(self.out_norm(h))


# ----------------------------------------------------------------------------
# DLinear + conditional diffusion on the residual
# ----------------------------------------------------------------------------
class Model(nn.Module):
    """
    DLinear-Diffusion

      y_hat = Linear_Trend(trend) + Linear_Seasonal(seasonal)   (original DLinear)
              + residual ~ p(r | history, y_dlinear)            (diffusion)

    where r = y_true - y_dlinear is the high-frequency part the linear model
    cannot explain. The diffusion model is conditioned on the history
    (its seasonal/high-frequency part) and on the DLinear forecast.

    I/O is identical to DLinear: forward(x[B, L, C]) -> [B, pred_len, C].

    Behaviour of forward():
      * training mode : returns the DLinear forecast only (cheap, so the
                        unchanged `MSE(model(x), y)` loop still works).
                        Set configs.sample_in_train=True to sample here too.
      * eval mode     : returns DLinear forecast + diffusion-sampled residual.
    For joint training use `compute_loss(x, y)` (see __main__ example).

    Optional configs (defaults in brackets):
      diff_steps[50] diff_hidden[128] diff_layers[2] ddim_steps[10]
      n_samples[1] diff_weight[1.0] sample_in_train[False] res_clip[5.0]
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # ---- original DLinear ------------------------------------------------
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for _ in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        # ---- diffusion head --------------------------------------------------
        g = lambda k, d: getattr(configs, k, d)
        self.T = g('diff_steps', 50)
        self.ddim_steps = min(g('ddim_steps', 10), self.T)
        self.n_samples = g('n_samples', 1)
        self.diff_weight = g('diff_weight', 1.0)
        self.sample_in_train = g('sample_in_train', False)
        self.res_clip = g('res_clip', 5.0)

        self.denoiser = ResidualDenoiser(
            self.seq_len, self.pred_len, self.channels,
            d=g('diff_hidden', 128), layers=g('diff_layers', 2))

        betas = torch.linspace(1e-4, 2e-2, self.T)
        self.register_buffer('alpha_bar', torch.cumprod(1.0 - betas, dim=0))
        # EMA of the std of the (history-normalised) residual -> unit-variance target
        self.register_buffer('res_std', torch.ones(1))

    # ------------------------------------------------------------------ DLinear
    def _dlinear(self, x):
        """Original DLinear. Returns forecast [B, H, C] and seasonal history [B, C, L]."""
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        base = (seasonal_output + trend_output).permute(0, 2, 1)  # [B, H, C]
        return base, seasonal_init

    # ---------------------------------------------------------------- condition
    def _condition(self, x, base, seasonal_hist):
        """
        Flatten to per-series vectors and build the (scale-free) condition.
        Returns cond [N, L+H], scale [N, 1], chan_idx [N]   with N = B*C.
        """
        B, L, C = x.shape
        xf = x.permute(0, 2, 1).reshape(B * C, L)
        mu = xf.mean(-1, keepdim=True)
        s = xf.std(-1, keepdim=True).detach() + 1e-5
        hist = seasonal_hist.reshape(B * C, L) / s
        bf = (base.detach().permute(0, 2, 1).reshape(B * C, self.pred_len) - mu) / s
        cond = torch.cat([hist, bf], dim=-1)
        chan_idx = torch.arange(C, device=x.device).repeat(B)
        return cond, s, chan_idx

    # ------------------------------------------------------------------ training
    def compute_loss(self, x, y):
        """
        x: [B, L, C] history, y: [B, pred_len, C] ground truth (slice y[:, -pred_len:] first).
        Returns total loss and a dict of its parts.
        The diffusion target is detached from DLinear, so the linear backbone is
        trained only by the MSE term and stays identical to vanilla DLinear.
        """
        base, seasonal_hist = self._dlinear(x)
        loss_base = F.mse_loss(base, y)

        cond, s, chan_idx = self._condition(x, base, seasonal_hist)
        B, H, C = y.shape
        r = (y - base).detach().permute(0, 2, 1).reshape(B * C, H) / s  # normalised residual

        if self.training:
            with torch.no_grad():
                self.res_std.mul_(0.99).add_(0.01 * r.std().clamp_min(1e-5))
        r0 = r / self.res_std

        t = torch.randint(0, self.T, (r0.size(0),), device=x.device)
        noise = torch.randn_like(r0)
        ab = self.alpha_bar[t][:, None]
        r_t = ab.sqrt() * r0 + (1 - ab).sqrt() * noise

        eps = self.denoiser(r_t, t, self.denoiser.encode_condition(cond, chan_idx))
        loss_diff = F.mse_loss(eps, noise)

        total = loss_base + self.diff_weight * loss_diff
        return total, {'base': loss_base.detach(), 'diff': loss_diff.detach()}

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def _sample_residual(self, cond, chan_idx, n_samples):
        """Deterministic DDIM sampling. Returns normalised residual [S, N, H]."""
        N = cond.size(0)
        cond_emb = self.denoiser.encode_condition(cond, chan_idx).repeat(n_samples, 1)
        r = torch.randn(n_samples * N, self.pred_len, device=cond.device)

        steps = torch.linspace(self.T - 1, 0, self.ddim_steps, device=cond.device).long()
        for i, t in enumerate(steps):
            ab_t = self.alpha_bar[t]
            ab_prev = self.alpha_bar[steps[i + 1]] if i + 1 < len(steps) else cond.new_tensor(1.0)
            tt = torch.full((r.size(0),), int(t), device=cond.device, dtype=torch.long)
            eps = self.denoiser(r, tt, cond_emb)
            r0 = ((r - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-self.res_clip, self.res_clip)
            eps = (r - ab_t.sqrt() * r0) / (1 - ab_t).sqrt().clamp_min(1e-8)
            r = ab_prev.sqrt() * r0 + (1 - ab_prev).sqrt() * eps
        return r.view(n_samples, N, self.pred_len)

    @torch.no_grad()
    def sample(self, x, n_samples=None):
        """Probabilistic forecasts: [S, B, pred_len, C] (DLinear + sampled residuals)."""
        n_samples = n_samples or self.n_samples
        B, _, C = x.shape
        base, seasonal_hist = self._dlinear(x)
        cond, s, chan_idx = self._condition(x, base, seasonal_hist)
        r = self._sample_residual(cond, chan_idx, n_samples) * self.res_std * s  # de-normalise
        r = r.view(n_samples, B, C, self.pred_len).permute(0, 1, 3, 2)
        return base.unsqueeze(0) + r

    # ------------------------------------------------------------------ forward
    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if self.training and not self.sample_in_train:
            return self._dlinear(x)[0]
        return self.sample(x).mean(0)  # [Batch, Output length, Channel]

