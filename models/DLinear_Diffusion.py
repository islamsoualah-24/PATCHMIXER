import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. كتل بناء DLinear الأصلية
# =============================================================================
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


# =============================================================================
# 2. تنفيذ Mamba بـ PyTorch الخالص (بدون الحاجة لتثبيت mamba-ssm)
# =============================================================================
class PurePyTorchMamba(nn.Module):
    """
    تنفيذ كامل لطبقة Mamba (Selective State Space Model) باستخدام PyTorch فقط.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def selective_scan(self, u, delta, A, B, C, D):
        """المعادلة التكرارية لـ Mamba (Selective Scan)"""
        b, l, d_in = u.shape
        n = A.shape[1]

        deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        deltaB_u = torch.einsum('bld,bln,bld->bldn', delta, B, u)

        x = torch.zeros((b, d_in, n), device=u.device, dtype=u.dtype)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = torch.einsum('bdn,bn->bd', x, C[:, i])
            ys.append(y)

        y = torch.stack(ys, dim=1)  # [b, l, d_in]
        y = y + u * D
        return y

    def forward(self, x):
        # x: [Batch, Length, d_model]
        b, l, _ = x.shape

        x_and_res = self.in_proj(x)
        x_in, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)

        # Causal Conv1D
        x_in = x_in.transpose(1, 2)
        x_in = self.conv1d(x_in)[:, :, :l]
        x_in = x_in.transpose(1, 2)
        x_in = F.silu(x_in)

        # SSM Parameters
        x_dbl = self.x_proj(x_in)
        delta, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        A = -torch.exp(self.A_log.float())

        # Selective Scan
        y = self.selective_scan(x_in, delta, A, B, C, self.D)
        y = y * F.silu(res)

        return self.out_proj(y)


# =============================================================================
# 3. شبكة إزالة الضجيج (Denoiser) باستخدام Mamba
# =============================================================================
def timestep_embedding(t, dim):
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


class MambaDiffusionBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mamba = PurePyTorchMamba(d_model=d, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.film = nn.Linear(d, 2 * d)

    def forward(self, h, c):
        scale, shift = self.film(c).unsqueeze(1).chunk(2, dim=-1)
        h_conditioned = self.norm(h) * (1 + scale) + shift
        return h + self.mamba(h_conditioned)


class ResidualDenoiserMamba(nn.Module):
    def __init__(self, seq_len, pred_len, channels, d=128, layers=2):
        super().__init__()
        self.d = d
        self.pred_len = pred_len
        self.in_proj = nn.Linear(1, d)

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

        self.blocks = nn.ModuleList([MambaDiffusionBlock(d) for _ in range(layers)])
        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, 1)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def encode_condition(self, cond, chan_idx):
        return self.cond_enc(cond) + self.chan_emb(chan_idx)

    def forward(self, r_t, t, cond_emb):
        c = cond_emb + self.t_mlp(timestep_embedding(t, self.d))
        r_t_seq = r_t.unsqueeze(-1)
        h = self.in_proj(r_t_seq)

        for block in self.blocks:
            h = block(h, c)

        h = self.out_norm(h)
        out = self.out_proj(h)
        return out.squeeze(-1)


# =============================================================================
# 4. النموذج الهجين الكامل: DLinear + Mamba Conditional Diffusion
# =============================================================================
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual

        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for _ in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        g = lambda key, default: getattr(configs, key, default)

        self.T = g('diff_steps', 50)
        self.ddim_steps = min(g('ddim_steps', 10), self.T)
        self.n_samples = g('n_samples', 50)
        self.diff_weight = g('diff_weight', 1.0)
        self.sample_in_train = g('sample_in_train', False)
        self.res_clip = g('res_clip', 5.0)
        self.d = g('diff_hidden', 128)

        self.denoiser = ResidualDenoiserMamba(
            self.seq_len,
            self.pred_len,
            self.channels,
            d=self.d,
            layers=g('diff_layers', 2)
        )

        betas = torch.linspace(1e-4, 2e-2, self.T)
        self.register_buffer('alpha_bar', torch.cumprod(1.0 - betas, dim=0))
        self.register_buffer('res_std', torch.ones(1))

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

        residual = (y - base).detach().permute(0, 2, 1).reshape(batch_size * channels, horizon) / scale

        if self.training:
            with torch.no_grad():
                current_std = residual.std().clamp_min(1e-5)
                self.res_std.mul_(0.99).add_(0.01 * current_std)

        residual_norm = residual / self.res_std

        t = torch.randint(0, self.T, (residual_norm.size(0),), device=x.device)
        noise = torch.randn_like(residual_norm)
        alpha_bar_t = self.alpha_bar[t][:, None]

        noisy_residual = alpha_bar_t.sqrt() * residual_norm + (1.0 - alpha_bar_t).sqrt() * noise

        cond_emb = self.denoiser.encode_condition(cond, chan_idx)
        predicted_noise = self.denoiser(noisy_residual, t, cond_emb)
        loss_diff = F.mse_loss(predicted_noise, noise)

        total_loss = loss_base + self.diff_weight * loss_diff

        return total_loss, {'base': loss_base.detach(), 'diff': loss_diff.detach()}

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
            alpha_bar_prev = self.alpha_bar[steps[i + 1]] if i + 1 < len(steps) else cond.new_tensor(1.0)

            t_batch = torch.full((residual.size(0),), int(timestep.item()), device=cond.device, dtype=torch.long)

            predicted_noise = self.denoiser(residual, t_batch, cond_emb)

            predicted_residual = (residual - (1.0 - alpha_bar_t).sqrt() * predicted_noise) / alpha_bar_t.sqrt()
            predicted_residual = predicted_residual.clamp(-self.res_clip, self.res_clip)
            predicted_noise = (residual - alpha_bar_t.sqrt() * predicted_residual) / (1.0 - alpha_bar_t).sqrt().clamp_min(1e-8)

            residual = alpha_bar_prev.sqrt() * predicted_residual + (1.0 - alpha_bar_prev).sqrt() * predicted_noise

        return residual.view(num_series, n_samples, self.pred_len).permute(1, 0, 2).contiguous()

    @torch.no_grad()
    def sample(self, x, n_samples=None):
        if n_samples is None:
            n_samples = self.n_samples

        batch_size, _, channels = x.shape

        base, seasonal_hist = self._dlinear(x)
        cond, scale, chan_idx = self._condition(x, base, seasonal_hist)

        residual = self._sample_residual(cond, chan_idx, n_samples)
        residual = residual * self.res_std * scale

        residual = residual.view(n_samples, batch_size, channels, self.pred_len).permute(0, 1, 3, 2).contiguous()

        return base.unsqueeze(0) + residual

    def forward(self, x, target=None, return_loss=False, loss_flag=0):
        if return_loss:
            if target is None:
                raise ValueError("target is required when return_loss=True")
            return self.compute_loss(x, target, loss_flag=loss_flag)

        if self.training and not self.sample_in_train:
            return self._dlinear(x)[0]

        return self.sample(x).mean(dim=0)
