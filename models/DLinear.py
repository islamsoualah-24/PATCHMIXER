import torch
import torch.nn as nn
import torch.fft

# ==========================================
# 1. مكون التردد المركب مع منع تسرب الطيف
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

        # شبكة زمنية [0 -> 1] لبناء جسر الحدود (تنتقل تلقائياً للـ GPU)
        self.register_buffer(
            "ramp_grid", 
            torch.linspace(0, 1, steps=seq_len).view(1, 1, seq_len)
        )

        # طبقة خفيفة لاستعادة أثر نقطتي البداية والنهاية في المستقبل بأمان
        self.boundary_proj = nn.Linear(2, pred_len)
        nn.init.zeros_(self.boundary_proj.weight)
        nn.init.zeros_(self.boundary_proj.bias)

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
        
        # 1. إزالة قفزة الحدود لجعل الإشارة دورية (x_periodic[0] == x_periodic[-1] == 0)
        x_start = x[:, :, 0:1]
        x_end = x[:, :, -1:]
        bridge = x_start + (x_end - x_start) * self.ramp_grid
        x_periodic = x - bridge
        
        # 2. تحويل فورييه السريع على الإشارة الخالية من تسرب الطيف
        low_specx = torch.fft.rfft(x_periodic, dim=-1)
        
        # 3. الإسقاط الخطي المركب
        if self.individual:
            low_specxy_ = torch.zeros(
                [low_specx.size(0), self.channels, int(self.dominance_freq * self.length_ratio)],
                dtype=low_specx.dtype, device=low_specx.device
            )
            for i in range(self.channels):
                low_specxy_[:, i, :] = self.freq_upsampler[i](low_specx[:, i, :])
        else:
            low_specxy_ = self.freq_upsampler(low_specx)

        # 4. حشو الترددات العالية بالأصفار
        out_freq_dim = self.total_len // 2 + 1
        low_specxy = torch.zeros(
            [low_specxy_.size(0), low_specxy_.size(1), out_freq_dim],
            dtype=low_specxy_.dtype, device=low_specxy_.device
        )
        min_len = min(low_specxy_.size(-1), out_freq_dim)
        low_specxy[:, :, :min_len] = low_specxy_[:, :, :min_len]

        # 5. التحويل العكسي وضبط الطاقة
        low_xy = torch.fft.irfft(low_specxy, n=self.total_len, dim=-1)
        low_xy = low_xy * self.length_ratio
        pred_seasonal = low_xy[:, :, -self.pred_len:]
        
        # 6. استعادة مساهمة الحدود في المستقبل
        endpoints = torch.cat([x_start, x_end], dim=-1)  # [Batch, channels, 2]
        boundary_pred = self.boundary_proj(endpoints)    # [Batch, channels, pred_len]
        
        return pred_seasonal + boundary_pred


# ==========================================
# 2. طبقة التقييس (RevIN) ومكون التفكيك
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
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
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
        return x.permute(0, 2, 1)

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# ==========================================
# 3. النموذج المدمج
# ==========================================
class Model(nn.Module): 
    def __init__(self, configs): 
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.individual = getattr(configs, 'individual', False)
        kernel_size = 25 
        
        self.revin = RevIN(num_features=self.enc_in)
        self.decomp = series_decomp(kernel_size)
        
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        
        self.Freq_Seasonal = ComplexFreqLinear(self.seq_len, self.pred_len, self.enc_in, self.individual)

    def forward(self, x):
        x = self.revin(x, 'norm')
        
        seasonal_init, trend_init = self.decomp(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        trend_output = self.Linear_Trend(trend_init)
        seasonal_output = self.Freq_Seasonal(seasonal_init)
        
        x_pred = (seasonal_output + trend_output).permute(0, 2, 1)
        x_pred = self.revin(x_pred, 'denorm')
        
        return x_pred
