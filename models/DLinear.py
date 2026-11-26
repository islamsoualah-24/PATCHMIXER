import torch
import torch.nn as nn
import torch.fft

# ==========================================
# 1. طبقة التقييس (RevIN)
# ==========================================
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            dim2reduce = tuple(range(1, x.ndim - 1))
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            x = x * self.stdev + self.mean
        else:
            raise NotImplementedError
        return x


# ==========================================
# 2. التفكيك متعدد المقاييس (Multi-Scale Decomposition)
# ==========================================
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)

class MultiScaleDecomp(nn.Module):
    def __init__(self, kernel_sizes=(25, 49, 97)):
        super(MultiScaleDecomp, self).__init__()
        self.moving_avgs = nn.ModuleList([moving_avg(k, stride=1) for k in kernel_sizes])
        self.scale_weights = nn.Parameter(torch.ones(len(kernel_sizes)) / len(kernel_sizes))

    def forward(self, x):
        weights = torch.softmax(self.scale_weights, dim=0)
        trend = sum(w * ma(x) for w, ma in zip(weights, self.moving_avgs))
        seasonal = x - trend
        return seasonal, trend


# ==========================================
# 3. خليط الخبراء الموجه ترددياً (Spectral-Guided MoLE)
# ==========================================
class SpectralMoLE(nn.Module):
    def __init__(self, seq_len, pred_len, num_experts=4, max_freq_bins=32, dropout=0.1, init_mode='mean'):
        super(SpectralMoLE, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_experts = num_experts
        self.init_mode = init_mode

        self.freq_bins = min(seq_len // 2 + 1, max_freq_bins)
        router_in_dim = self.freq_bins * 2
        router_hidden = max(router_in_dim // 2, 16)

        self.router_norm = nn.LayerNorm(router_in_dim)
        self.router = nn.Sequential(
            nn.Linear(router_in_dim, router_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden, num_experts)
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

        self.experts = nn.Linear(seq_len, pred_len * num_experts)
        self._init_experts()
        self.last_routing_weights = None

    def _init_experts(self):
        with torch.no_grad():
            if self.init_mode == 'zero':
                base_w = torch.zeros(self.num_experts, self.pred_len, self.seq_len)
            else:
                base_w = (1.0 / self.seq_len) * torch.ones(self.num_experts, self.pred_len, self.seq_len)

            noise = torch.randn_like(base_w) * (1e-2 / self.seq_len)
            noise = noise - noise.mean(dim=0, keepdim=True)
            self.experts.weight.copy_((base_w + noise).view(self.num_experts * self.pred_len, self.seq_len))
            self.experts.bias.zero_()

    def forward(self, x):
        # x: [B, C, seq_len]
        B, C, _ = x.shape

        x_fft = torch.fft.rfft(x, dim=-1)[:, :, :self.freq_bins]
        spec_feat = self.router_norm(torch.cat([x_fft.real, x_fft.imag], dim=-1))

        weights = torch.softmax(self.router(spec_feat), dim=-1)
        self.last_routing_weights = weights.detach()

        all_preds = self.experts(x).view(B, C, self.num_experts, self.pred_len)
        return torch.sum(all_preds * weights.unsqueeze(-1), dim=2)


# ==========================================
# 4. النموذج المطور (Anchored Multi-Scale Spectral-MoLE)
# ==========================================
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        enc_in = configs.enc_in

        num_experts = getattr(configs, 'num_experts', 4)
        dropout = getattr(configs, 'dropout', 0.1)
        kernel_sizes = getattr(configs, 'kernel_sizes', (25, 49, 97))

        self.revin = RevIN(num_features=enc_in)
        self.decomp = MultiScaleDecomp(kernel_sizes=kernel_sizes)

        # 1. خبير الموسمية السريعة
        self.MoLE_Seasonal = SpectralMoLE(
            self.seq_len, self.pred_len, num_experts=num_experts, dropout=dropout, init_mode='mean'
        )

        # 2. خبير الاتجاه المرسّخ (يتنبأ بالتغير النسبي حول آخر قيمة اتجاه)
        self.MoLE_Trend = SpectralMoLE(
            self.seq_len, self.pred_len, num_experts=num_experts, dropout=dropout, init_mode='zero'
        )

        # 3. وحدة تصحيح إزاحة المستوى (تلتقط تحول المتوسط الحديث مقابل المتوسط الكلي)
        self.recent_len = min(24, self.seq_len)
        self.shift_proj = nn.Linear(2, self.pred_len)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)

    def forward(self, x):
        # x: [Batch, seq_len, channels]
        x = self.revin(x, 'norm')

        seasonal_init, trend_init = self.decomp(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)  # [B, C, L]
        trend_init = trend_init.permute(0, 2, 1)        # [B, C, L]

        # التنبؤ بالموسمية
        seasonal_output = self.MoLE_Seasonal(seasonal_init)

        # ترسيخ الاتجاه بآخر قيمة لتجنب التراجع نحو المتوسط التاريخي
        last_trend = trend_init[:, :, -1:]  # [B, C, 1]
        trend_output = last_trend + self.MoLE_Trend(trend_init - last_trend)

        # حساب إزاحة المستوى الحديث (الفرق بين آخر 24 خطوة ومتوسط النافذة)
        recent_mean = x[:, -self.recent_len:, :].mean(dim=1).permute(0, 2, 1) if x.ndim == 4 else x[:, -self.recent_len:, :].mean(dim=1).unsqueeze(-1)
        recent_slope = (x[:, -1, :] - x[:, -self.recent_len, :]).unsqueeze(-1)
        shift_feat = torch.cat([recent_mean, recent_slope], dim=-1)  # [B, C, 2]
        shift_output = self.shift_proj(shift_feat)                   # [B, C, pred_len]

        x_pred = (seasonal_output + trend_output + shift_output).permute(0, 2, 1)
        x_pred = self.revin(x_pred, 'denorm')

        return x_pred
