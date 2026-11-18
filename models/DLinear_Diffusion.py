import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Peak-Sensitive Loss for Amplitude Correction
# -----------------------------------------------------------------------------
class PeakSensitiveLoss(nn.Module):
    """
    دالة تكلفة تعطي وزناً أكبر للأخطاء عند القيم المتطرفة (القمم والقيعان).
    """
    def __init__(self, alpha=2.0):
        super(PeakSensitiveLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        base_loss = self.mse(pred, target)
        
        # إعطاء وزن بناءً على القيمة المطلقة للهدف
        # القيم الأبعد عن الصفر ستحصل على عقاب أكبر عند الخطأ
        weights = 1.0 + self.alpha * torch.abs(target)
        
        weighted_loss = base_loss * weights
        return weighted_loss.mean()


# -----------------------------------------------------------------------------
# Original DLinear building blocks
# -----------------------------------------------------------------------------
class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series."""

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=stride, padding=0
        )

    def forward(self, x):
        front = x[:, 0:1, :].repeat(
            1, (self.kernel_size - 1) // 2, 1
        )
        end = x[:, -1:, :].repeat(
            1, (self.kernel_size - 1) // 2, 1
        )
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    """Series decomposition block."""

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# -----------------------------------------------------------------------------
# Diffusion timestep embedding
# -----------------------------------------------------------------------------
def timestep_embedding(t, dim):
    """Sinusoidal embedding: t [N] -> [N, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        - torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half, 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLMBlock(nn.Module):
    """Residual MLP block modulated by timestep and history condition."""

    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.film = nn.Linear(d, 2 * d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, d)
        )

    def forward(self, h, c):
        scale, shift = self.film(c).chunk(2, dim=-1)
        return h + self.mlp(self.norm(h) * (1 + scale) + shift)


class ResidualDenoiser(nn.Module):
    """
    Predict epsilon_theta(r_t, t | history, base forecast).
    Input/output vectors have shape [N=B*C, pred_len].
    The denoiser is shared across channels.
    """

    def __init__(self, seq_len, pred_len, channels, d=128, layers=2):
        super().__init__()
        self.d = d
        self.in_proj = nn.Linear(pred_len, d)
        self.cond_enc = nn.Sequential(
            nn.Linear(seq_len + pred_len, d),
            nn.GELU(),
            nn.Linear(d, d)
        )
        self.chan_emb = nn.Embedding(channels, d)
        self.t_mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d)
        )
        self.blocks = nn.ModuleList(
            [FiLMBlock(d) for _ in range(layers)]
        )
        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, pred_len)

        # Start by predicting zero noise.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def encode_condition(self, cond, chan_idx):
        return self.cond_enc(cond) + self.chan_emb(chan_idx)

    def forward(self, r_t, t, cond_emb):
        c = cond_emb + self.t_mlp(timestep_embedding(t, self.d))
        h = self.in_proj(r_t)
        for block in self.blocks:
            h = block(h, c)
        return self.out_proj(self.out_norm(h))


# -----------------------------------------------------------------------------
# DLinear + conditional diffusion
# -----------------------------------------------------------------------------
class Model(nn.Module):
    """
    DLinear-Diffusion.

    Forecast = DLinear forecast + diffusion-generated residual.

    Input:
        x: [B, seq_len, C]

    Standard output:
        [B, pred_len, C]

    Joint training:
        model(x, target=y, return_loss=True, loss_flag=...)
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Original DLinear
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for _ in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len)
                )
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len)
                )
        else:
            self.Linear_Seasonal = nn.Linear(
                self.seq_len, self.pred_len
            )
            self.Linear_Trend = nn.Linear(
                self.seq_len, self.pred_len
            )

        # Diffusion configuration
        g = lambda key, default: getattr(configs, key, default)

        self.T = g('diff_steps', 50)
        self.ddim_steps = min(g('ddim_steps', 10), self.T)
        self.n_samples = g('n_samples', 50)
        self.diff_weight = g('diff_weight', 1.0)
        self.sample_in_train = g('sample_in_train', False)
        self.res_clip = g('res_clip', 5.0)

        # Fix: define self.d on Model; sampling uses this attribute.
        self.d = g('diff_hidden', 128)

        self.denoiser = ResidualDenoiser(
            self.seq_len,
            self.pred_len,
            self.channels,
            d=self.d,
            layers=g('diff_layers', 2)
        )

        betas = torch.linspace(1e-4, 2e-2, self.T)
        self.register_buffer(
            'alpha_bar',
            torch.cumprod(1.0 - betas, dim=0)
        )
        self.register_buffer('res_std', torch.ones(1))

        # --- Peak Sensitive Loss ---
        self.peak_loss = PeakSensitiveLoss(alpha=2.0)

    # -------------------------------------------------------------------------
    # DLinear forecast
    # -------------------------------------------------------------------------
    def _dlinear(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros(
                seasonal_init.size(0),
                seasonal_init.size(1),
                self.pred_len,
                dtype=seasonal_init.dtype,
                device=seasonal_init.device
            )
            trend_output = torch.zeros(
                trend_init.size(0),
                trend_init.size(1),
                self.pred_len,
                dtype=trend_init.dtype,
                device=trend_init.device
            )
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :]
                )
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :]
                )
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        base = (seasonal_output + trend_output).permute(0, 2, 1)
        return base, seasonal_init

    # -------------------------------------------------------------------------
    # Build conditioning information
    # -------------------------------------------------------------------------
    def _condition(self, x, base, seasonal_hist):
        batch_size, seq_len, channels = x.shape

        xf = x.permute(0, 2, 1).reshape(
            batch_size * channels, seq_len
        )
        mu = xf.mean(-1, keepdim=True)
        scale = xf.std(-1, keepdim=True).detach() + 1e-5

        hist = seasonal_hist.reshape(
            batch_size * channels, seq_len
        ) / scale

        base_flat = base.detach().permute(0, 2, 1).reshape(
            batch_size * channels, self.pred_len
        )
        base_norm = (base_flat - mu) / scale
        cond = torch.cat([hist, base_norm], dim=-1)

        chan_idx = torch.arange(
            channels, device=x.device
        ).repeat(batch_size)

        return cond, scale, chan_idx

    # -------------------------------------------------------------------------
    # Joint training loss
    # -------------------------------------------------------------------------
    def compute_loss(self, x, y, loss_flag=0):
        base, seasonal_hist = self._dlinear(x)

        if loss_flag == 1:
            loss_base = F.l1_loss(base, y)
        elif loss_flag == 2:
            loss_base = (
                0.5 * F.l1_loss(base, y)
                + 0.5 * F.mse_loss(base, y)
            )
        elif loss_flag == 3:
            loss_base = F.smooth_l1_loss(base, y)
        elif loss_flag == 4:
            # استخدام دالة التكلفة الحساسة للقمم
            loss_base = self.peak_loss(base, y)
        else:
            loss_base = F.mse_loss(base, y)

        cond, scale, chan_idx = self._condition(
            x, base, seasonal_hist
        )
        batch_size, horizon, channels = y.shape

        residual = (
            (y - base).detach()
            .permute(0, 2, 1)
            .reshape(batch_size * channels, horizon)
            / scale
        )

        if self.training:
            with torch.no_grad():
                current_std = residual.std().clamp_min(1e-5)
                self.res_std.mul_(0.99).add_(0.01 * current_std)

        residual_norm = residual / self.res_std

        t = torch.randint(
            0, self.T, (residual_norm.size(0),), device=x.device
        )
        noise = torch.randn_like(residual_norm)
        alpha_bar_t = self.alpha_bar[t][:, None]

        noisy_residual = (
            alpha_bar_t.sqrt() * residual_norm
            + (1.0 - alpha_bar_t).sqrt() * noise
        )

        cond_emb = self.denoiser.encode_condition(cond, chan_idx)
        predicted_noise = self.denoiser(
            noisy_residual, t, cond_emb
        )
        loss_diff = F.mse_loss(predicted_noise, noise)

        total_loss = loss_base + self.diff_weight * loss_diff

        return total_loss, {
            'base': loss_base.detach(),
            'diff': loss_diff.detach()
        }

    # -------------------------------------------------------------------------
    # DDIM residual sampling
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _sample_residual(self, cond, chan_idx, n_samples):
        """
        Returns residual samples with shape [S, N, pred_len].
        """
        num_series = cond.size(0)

        cond_emb = self.denoiser.encode_condition(cond, chan_idx)
        cond_emb = cond_emb.repeat_interleave(n_samples, dim=0)

        residual = torch.randn(
            num_series * n_samples,
            self.pred_len,
            device=cond.device,
            dtype=cond.dtype
        )

        steps = torch.linspace(
            self.T - 1,
            0,
            self.ddim_steps,
            device=cond.device
        ).long()

        for i, timestep in enumerate(steps):
            alpha_bar_t = self.alpha_bar[timestep]

            if i + 1 < len(steps):
                alpha_bar_prev = self.alpha_bar[steps[i + 1]]
            else:
                alpha_bar_prev = cond.new_tensor(1.0)

            t_batch = torch.full(
                (residual.size(0),),
                int(timestep.item()),
                device=cond.device,
                dtype=torch.long
            )

            assert cond_emb.shape == (
                residual.size(0), self.d
            ), (
                f"Condition shape error: {cond_emb.shape}; "
                f"expected {(residual.size(0), self.d)}"
            )

            predicted_noise = self.denoiser(
                residual, t_batch, cond_emb
            )

            predicted_residual = (
                residual
                - (1.0 - alpha_bar_t).sqrt() * predicted_noise
            ) / alpha_bar_t.sqrt()

            predicted_residual = predicted_residual.clamp(
                -self.res_clip, self.res_clip
            )

            predicted_noise = (
                residual - alpha_bar_t.sqrt() * predicted_residual
            ) / (1.0 - alpha_bar_t).sqrt().clamp_min(1e-8)

            residual = (
                alpha_bar_prev.sqrt() * predicted_residual
                + (1.0 - alpha_bar_prev).sqrt() * predicted_noise
            )

        # Correct ordering for repeat_interleave:
        # [N, S, H] -> [S, N, H].
        return (
            residual
            .view(num_series, n_samples, self.pred_len)
            .permute(1, 0, 2)
            .contiguous()
        )

    @torch.no_grad()
    def sample(self, x, n_samples=None):
        """Return forecasts with shape [S, B, pred_len, C]."""
        if n_samples is None:
            n_samples = self.n_samples

        batch_size, _, channels = x.shape

        base, seasonal_hist = self._dlinear(x)
        cond, scale, chan_idx = self._condition(
            x, base, seasonal_hist
        )

        residual = self._sample_residual(
            cond, chan_idx, n_samples
        )
        residual = residual * self.res_std * scale

        residual = (
            residual
            .view(
                n_samples,
                batch_size,
                channels,
                self.pred_len
            )
            .permute(0, 1, 3, 2)
            .contiguous()
        )

        return base.unsqueeze(0) + residual

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    def forward(
        self,
        x,
        target=None,
        return_loss=False,
        loss_flag=0
    ):
        if return_loss:
            if target is None:
                raise ValueError(
                    "target is required when return_loss=True"
                )
            return self.compute_loss(
                x, target, loss_flag=loss_flag
            )

        if self.training and not self.sample_in_train:
            return self._dlinear(x)[0]

        return self.sample(x).mean(dim=0)
