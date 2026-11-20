import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. DLinear building blocks (التوقع الأساسي)
# -----------------------------------------------------------------------------
class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series."""
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
    """Series decomposition block."""
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# -----------------------------------------------------------------------------
# 2. Diffusion Timestep Embedding
# -----------------------------------------------------------------------------
def timestep_embedding(t, dim):
    """Sinusoidal embedding: t [N] -> [N, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        - torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# -----------------------------------------------------------------------------
# 3. NEW: Transformer-Based Residual Denoiser (التحسين الجديد)
# -----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """تضمين المواقع لتمكين المحول من فهم التسلسل الزمني"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # [1, max_len, d_model]

    def forward(self, x):
        # x shape: [Batch, Seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

class TransformerResidualDenoiser(nn.Module):
    """
    شبكة إزالة الضجيج تعتمد على Transformers بدلاً من MLP.
    Predict epsilon_theta(r_t, t | history, base forecast).
    """
    def __init__(self, seq_len, pred_len, channels, d=128, nheads=4, layers=2):
        super().__init__()
        self.d = d
        self.pred_len = pred_len
        
        # تشفير الشرط (Condition)
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
        
        # تحضير السلسلة للمحول
        self.in_proj = nn.Linear(1, d)
        self.pos_encoder = PositionalEncoding(d_model=d, max_len=pred_len)
        
        # طبقات المحول (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, 
            nhead=nheads, 
            dim_feedforward=d * 2, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        # طبقة الإخراج
        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, 1)
        
        # تهيئة بأصفار لبدء مستقر
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def encode_condition(self, cond, chan_idx):
        return self.cond_enc(cond) + self.chan_emb(chan_idx)

    def forward(self, r_t, t, cond_emb):
        # r_t shape: [N, pred_len]
        
        c = cond_emb + self.t_mlp(timestep_embedding(t, self.d))
        c = c.unsqueeze(1) # [N, 1, d]
        
        # تحويل البواقي إلى سلسلة زمنية
        x = r_t.unsqueeze(-1) # [N, pred_len, 1]
        x = self.in_proj(x)   # [N, pred_len, d]
        
        x = self.pos_encoder(x)
        x = x + c # إضافة التوجيه
        
        out = self.transformer_encoder(x) # [N, pred_len, d]
        
        out = self.out_norm(out)
        out = self.out_proj(out).squeeze(-1) # [N, pred_len]
        
        return out


# -----------------------------------------------------------------------------
# 4. Model (DLinear + Conditional Diffusion)
# -----------------------------------------------------------------------------
class Model(nn.Module):
    """
    DLinear-Diffusion with Transformer Denoiser.
    Forecast = DLinear forecast + diffusion-generated residual.
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
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        # Diffusion configuration
        g = lambda key, default: getattr(configs, key, default)

        self.T = g('diff_steps', 50)
        self.ddim_steps = min(g('ddim_steps', 10), self.T)
        self.n_samples = g('n_samples', 50)
        self.diff_weight = g('diff_weight', 1.0)
        self.sample_in_train = g('sample_in_train', False)
        self.res_clip = g('res_clip', 5.0)
        self.d = g('diff_hidden', 128)

        # استدعاء المحول الجديد بدلاً من الشبكة القديمة
        self.denoiser = TransformerResidualDenoiser(
            self.seq_len,
            self.pred_len,
            self.channels,
            d=self.d,
            nheads=4,
            layers=g('diff_layers', 2)
        )

        betas = torch.linspace(1e-4, 2e-2, self.T)
        self.register_buffer('alpha_bar', torch.cumprod(1.0 - betas, dim=0))
        self.register_buffer('res_std', torch.ones(1))

    # -------------------------------------------------------------------------
    # DLinear forecast
    # -------------------------------------------------------------------------
    def _dlinear(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros(
                seasonal_init.size(0), seasonal_init.size(1), self.pred_len,
                dtype=seasonal_init.dtype, device=seasonal_init.device
            )
            trend_output = torch.zeros(
                trend_init.size(0), trend_init.size(1), self.pred_len,
                dtype=trend_init.dtype, device=trend_init.device
            )
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
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

        xf = x.permute(0, 2, 1).reshape(batch_size * channels, seq_len)
        mu = xf.mean(-1, keepdim=True)
        scale = xf.std(-1, keepdim=True).detach() + 1e-5

        hist = seasonal_hist.reshape(batch_size * channels, seq_len) / scale

        base_flat = base.detach().permute(0, 2, 1).reshape(batch_size * channels, self.pred_len)
        base_norm = (base_flat - mu) / scale
        cond = torch.cat([hist, base_norm], dim=-1)

        chan_idx = torch.arange(channels, device=x.device).repeat(batch_size)

        return cond, scale, chan_idx

    # -------------------------------------------------------------------------
    # Joint training loss
    # -------------------------------------------------------------------------
    def compute_loss(self, x, y, loss_flag=0):
        base, seasonal_hist = self._dlinear(x)

        if loss_flag == 1:
            loss_base = F.l1_loss(base, y)
        elif loss_flag == 2:
            loss_base = 0.5 * F.l1_loss(base, y) + 0.5 * F.mse_loss(base, y)
        elif loss_flag == 3:
            loss_base = F.smooth_l1_loss(base, y)
        else:
            loss_base = F.mse_loss(base, y)

        cond, scale, chan_idx = self._condition(x, base, seasonal_hist)
        batch_size, horizon, channels = y.shape

        residual = (
            (y - base).detach()
            .permute(0, 2, 1)
            .reshape(batch_size * channels, horizon) / scale
        )

        if self.training:
            with torch.no_grad():
                current_std = residual.std().clamp_min(1e-5)
                self.res_std.mul_(0.99).add_(0.01 * current_std)

        residual_norm = residual / self.res_std

        t = torch.randint(0, self.T, (residual_norm.size(0),), device=x.device)
        noise = torch.randn_like(residual_norm)
        alpha_bar_t = self.alpha_bar[t][:, None]

        noisy_residual = (
            alpha_bar_t.sqrt() * residual_norm
            + (1.0 - alpha_bar_t).sqrt() * noise
        )

        cond_emb = self.denoiser.encode_condition(cond, chan_idx)
        predicted_noise = self.denoiser(noisy_residual, t, cond_emb)
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
        num_series = cond.size(0)

        cond_emb = self.denoiser.encode_condition(cond, chan_idx)
        cond_emb = cond_emb.repeat_interleave(n_samples, dim=0)

        residual = torch.randn(
            num_series * n_samples, self.pred_len,
            device=cond.device, dtype=cond.dtype
        )

        steps = torch.linspace(self.T - 1, 0, self.ddim_steps, device=cond.device).long()

        for i, timestep in enumerate(steps):
            alpha_bar_t = self.alpha_bar[timestep]

            if i + 1 < len(steps):
                alpha_bar_prev = self.alpha_bar[steps[i + 1]]
            else:
                alpha_bar_prev = cond.new_tensor(1.0)

            t_batch = torch.full(
                (residual.size(0),), int(timestep.item()),
                device=cond.device, dtype=torch.long
            )

            predicted_noise = self.denoiser(residual, t_batch, cond_emb)

            predicted_residual = (
                residual - (1.0 - alpha_bar_t).sqrt() * predicted_noise
            ) / alpha_bar_t.sqrt()

            predicted_residual = predicted_residual.clamp(-self.res_clip, self.res_clip)

            predicted_noise = (
                residual - alpha_bar_t.sqrt() * predicted_residual
            ) / (1.0 - alpha_bar_t).sqrt().clamp_min(1e-8)

            residual = (
                alpha_bar_prev.sqrt() * predicted_residual
                + (1.0 - alpha_bar_prev).sqrt() * predicted_noise
            )

        return (
            residual
            .view(num_series, n_samples, self.pred_len)
            .permute(1, 0, 2)
            .contiguous()
        )

    @torch.no_grad()
    def sample(self, x, n_samples=None):
        if n_samples is None:
            n_samples = self.n_samples

        batch_size, _, channels = x.shape

        base, seasonal_hist = self._dlinear(x)
        cond, scale, chan_idx = self._condition(x, base, seasonal_hist)

        residual = self._sample_residual(cond, chan_idx, n_samples)
        residual = residual * self.res_std * scale

        residual = (
            residual
            .view(n_samples, batch_size, channels, self.pred_len)
            .permute(0, 1, 3, 2)
            .contiguous()
        )

        return base.unsqueeze(0) + residual

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    def forward(self, x, target=None, return_loss=False, loss_flag=0):
        if return_loss:
            if target is None:
                raise ValueError("target is required when return_loss=True")
            return self.compute_loss(x, target, loss_flag=loss_flag)

        if self.training and not self.sample_in_train:
            return self._dlinear(x)[0]

        return self.sample(x).mean(dim=0)
