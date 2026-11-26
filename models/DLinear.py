import torch
import torch.nn as nn
import torch.fft

# ==========================================
# 1. مكون خلط القنوات المركب (فهم العلاقات بين المتغيرات)
# ==========================================
class ComplexChannelMixer(nn.Module):
    def __init__(self, channels, reduction=2, dropout=0.1):
        super(ComplexChannelMixer, self).__init__()
        # استخدام عنق زجاجة (Bottleneck) لتقليل المعاملات ومنع Overfitting
        hidden_dim = max(channels // reduction, 8)
        
        self.fc1 = nn.Linear(channels, hidden_dim).to(torch.cfloat)
        self.fc2 = nn.Linear(hidden_dim, channels).to(torch.cfloat)
        self.dropout = nn.Dropout(dropout)
        
        # معامل البوابة يبدأ من 0.0 لضمان استقرار التدريب (Zero-Init Residual)
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1))

    def complex_gelu(self, z):
        # تطبيق دالة التنشيط على الجزء الحقيقي والتخيلي بشكل منفصل
        return torch.complex(
            nn.functional.gelu(z.real),
            nn.functional.gelu(z.imag)
        )

    def complex_dropout(self, z):
        # تطبيق Dropout بنفس القناع على الحقيقي والتخيلي للحفاظ على الطور
        if not self.training:
            return z
        mask = torch.ones_like(z.real)
        mask = self.dropout(mask)
        return z * mask

    def forward(self, x_freq):
        # x_freq shape: [Batch, channels, freq_dim] (Complex)
        
        # نقل القنوات للبعد الأخير لتطبيق الطبقات الخطية عليها
        x_perm = x_freq.permute(0, 2, 1)  # [Batch, freq_dim, channels]
        
        out = self.fc1(x_perm)
        out = self.complex_gelu(out)
        out = self.complex_dropout(out)
        out = self.fc2(out)
        
        out = out.permute(0, 2, 1)  # [Batch, channels, freq_dim]
        
        # الدمج التكيفي مع الاتصال المتبقي
        return x_freq + self.alpha * out


# ==========================================
# 2. مكون التردد المركب مع خيار خلط القنوات
# ==========================================
class ComplexFreqLinear(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, individual=False, use_channel_mixing=True):
        super(ComplexFreqLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len
        self.individual = individual
        self.channels = enc_in
        self.use_channel_mixing = use_channel_mixing and (enc_in > 1)
        
        self.dominance_freq = seq_len // 2 + 1
        self.length_ratio = self.total_len / self.seq_len

        # إضافة وحدة فهم العلاقات بين المتغيرات
        if self.use_channel_mixing:
            self.channel_mixer = ComplexChannelMixer(enc_in)

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
        
        # 1. تحويل فورييه السريع
        low_specx = torch.fft.rfft(x, dim=-1)
        
        # 2. التقاط العلاقات بين المتغيرات في المجال الترددي قبل التوسيع
        if self.use_channel_mixing:
            low_specx = self.channel_mixer(low_specx)
        
        # 3. الإسقاط الخطي المركب لتوسيع الطيف
        if self.individual:
            low_specxy_ = torch.zeros(
                [low_specx.size(0), self.channels, int(self.dominance_freq * self.length_ratio)],
                dtype=low_specx.dtype, device=low_specx.device
            )
            for i in range(self.channels):
                low_specxy_[:, i, :] = self.freq_upsampler[i](low_specx[:, i, :])
        else:
            low_specxy_ = self.freq_upsampler(low_specx)

        # 4. حشو الترددات العالية بالأصفار (Zero-Padding)
        out_freq_dim = self.total_len // 2 + 1
        low_specxy = torch.zeros(
            [low_specxy_.size(0), low_specxy_.size(1), out_freq_dim],
            dtype=low_specxy_.dtype, device=low_specxy_.device
        )
        min_len = min(low_specxy_.size(-1), out_freq_dim)
        low_specxy[:, :, :min_len] = low_specxy_[:, :, :min_len]

        # 5. التحويل العكسي واستخراج المستقبل
        low_xy = torch.fft.irfft(low_specxy, n=self.total_len, dim=-1)
        low_xy = low_xy * self.length_ratio
        
        return low_xy[:, :, -self.pred_len:]


# ==========================================
# 3. طبقة التقييس (RevIN) ومكون التفكيك
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
# 4. النموذج المدمج النهائي
# ==========================================
class Model(nn.Module): 
    def __init__(self, configs): 
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.individual = getattr(configs, 'individual', False)
        self.use_channel_mixing = getattr(configs, 'use_channel_mixing', True)
        kernel_size = 25 
        
        self.revin = RevIN(num_features=self.enc_in)
        self.decomp = series_decomp(kernel_size)
        
        # الاتجاه في المجال الزمني
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        
        # الموسمية في المجال الترددي المركب + فهم العلاقات بين المتغيرات
        self.Freq_Seasonal = ComplexFreqLinear(
            self.seq_len, self.pred_len, self.enc_in, 
            individual=self.individual, 
            use_channel_mixing=self.use_channel_mixing
        )

    def forward(self, x):
        # x: [Batch, seq_len, channels]
        x = self.revin(x, 'norm')
        
        seasonal_init, trend_init = self.decomp(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        trend_output = self.Linear_Trend(trend_init)
        seasonal_output = self.Freq_Seasonal(seasonal_init)
        
        x_pred = (seasonal_output + trend_output).permute(0, 2, 1)
        x_pred = self.revin(x_pred, 'denorm')
        
        return x_pred
