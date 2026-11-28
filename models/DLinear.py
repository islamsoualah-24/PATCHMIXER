import torch
import torch.nn as nn
import torch.fft

# ==========================================
# 1. طبقة التقييس العكسي (RevIN)
# ==========================================
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


# ==========================================
# 2. مكون التفكيك (Series Decomposition)
# ==========================================
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# ==========================================
# 3. مكون التردد المركب (Complex Frequency Upsampling)
# ==========================================
class ComplexFreqLinear(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, individual=False):
        super(ComplexFreqLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len
        self.individual = individual
        self.channels = enc_in

        self.dominance_freq = seq_len // 2 + 1
        self.length_ratio = self.total_len / self.seq_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList([
                nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(torch.cfloat)
                for _ in range(self.channels)
            ])
        else:
            self.freq_upsampler = nn.Linear(
                self.dominance_freq, int(self.dominance_freq * self.length_ratio)
            ).to(torch.cfloat)

    def forward(self, x):
        # x shape: [Batch, channels, seq_len]
        low_specx = torch.fft.rfft(x, dim=-1)

        if self.individual:
            low_specxy_ = torch.zeros(
                [low_specx.size(0), self.channels, int(self.dominance_freq * self.length_ratio)],
                dtype=low_specx.dtype, device=low_specx.device
            )
            for i in range(self.channels):
                low_specxy_[:, i, :] = self.freq_upsampler[i](low_specx[:, i, :])
        else:
            low_specxy_ = self.freq_upsampler(low_specx)

        out_freq_dim = self.total_len // 2 + 1
        low_specxy = torch.zeros(
            [low_specxy_.size(0), low_specxy_.size(1), out_freq_dim],
            dtype=low_specxy_.dtype, device=low_specxy_.device
        )
        min_len = min(low_specxy_.size(-1), out_freq_dim)
        low_specxy[:, :, :min_len] = low_specxy_[:, :, :min_len]

        low_xy = torch.fft.irfft(low_specxy, n=self.total_len, dim=-1)
        low_xy = low_xy * self.length_ratio

        pred_seasonal = low_xy[:, :, -self.pred_len:]
        return pred_seasonal


# ==========================================
# 4. كتلة رونج-كوتا الخفيفة بين القنوات (Channel-Mixing RK4)
# ==========================================
class ChannelODEFunc(nn.Module):
    def __init__(self, enc_in, hidden_dim=16, dropout=0.3):
        super(ChannelODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, enc_in)
        )
        nn.init.normal_(self.net[-1].weight, mean=0, std=0.01)
        nn.init.constant_(self.net[-1].bias, val=0)

    def forward(self, h):
        return self.net(h)


class ChannelRK4Block(nn.Module):
    def __init__(self, enc_in, hidden_dim=16, num_steps=4, dropout=0.3):
        super(ChannelRK4Block, self).__init__()
        self.ode_func = ChannelODEFunc(enc_in, hidden_dim, dropout)
        self.num_steps = num_steps
        self.dt = 1.0 / num_steps

    def forward(self, x):
        # x shape: [Batch, pred_len, enc_in]
        h = x
        for _ in range(self.num_steps):
            k1 = self.ode_func(h)
            k2 = self.ode_func(h + 0.5 * self.dt * k1)
            k3 = self.ode_func(h + 0.5 * self.dt * k2)
            k4 = self.ode_func(h + self.dt * k3)
            h = h + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return h


# ==========================================
# 5. النموذج الكامل (Cascaded Freq-RK4 DLinear + RevIN)
# ==========================================
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.individual = getattr(configs, 'individual', False)
        kernel_size = 25

        rk4_hidden = getattr(configs, 'rk4_hidden', 16)
        rk4_steps = getattr(configs, 'rk4_steps', 4)
        rk4_dropout = getattr(configs, 'dropout', 0.3)

        # 1. التقييس والتفكيك
        self.revin = RevIN(num_features=self.enc_in)
        self.decomp = series_decomp(kernel_size)

        # 2. مسار الاتجاه (خطي بالكامل)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
        )

        # 3. مسار الدورية التسلسلي: تردد مركب (زمن) -> ثم RK4 (قنوات)
        self.Freq_Seasonal = ComplexFreqLinear(
            self.seq_len, self.pred_len, self.enc_in, self.individual
        )
        self.channel_rk4 = ChannelRK4Block(
            enc_in=self.enc_in,
            hidden_dim=rk4_hidden,
            num_steps=rk4_steps,
            dropout=rk4_dropout
        )

    def forward(self, x):
        # x shape: [Batch, seq_len, channels]
        x = self.revin(x, 'norm')

        seasonal_init, trend_init = self.decomp(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)  # [Batch, channels, seq_len]
        trend_init = trend_init.permute(0, 2, 1)        # [Batch, channels, seq_len]

        # أ. الاتجاه (خطي)
        trend_output = self.Linear_Trend(trend_init).permute(0, 2, 1)  # [Batch, pred_len, channels]

        # ب. الدورية (تنبؤ في المجال الترددي المركب ثم صقل ديناميكي بين القنوات بـ RK4)
        seasonal_output = self.Freq_Seasonal(seasonal_init).permute(0, 2, 1)
        seasonal_output = self.channel_rk4(seasonal_output)

        # ج. الدمج النهائي والتقييس العكسي
        x_pred = seasonal_output + trend_output
        x_pred = self.revin(x_pred, 'denorm')

        return x_pred
