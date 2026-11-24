import torch
import torch.nn as nn
import torch.fft # ضروري لتحويلات فورييه

# ==========================================
# 1. مكون التردد الكامل (بدون اقتطاع)
# ==========================================
class ComplexFreqLinear(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, individual=False):
        super(ComplexFreqLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len
        self.individual = individual
        self.channels = enc_in
        
        # الاحتفاظ بكامل الطيف الترددي الذي أعطى النتائج الممتازة سابقاً
        self.dominance_freq = seq_len // 2 + 1
        self.length_ratio = self.total_len / self.seq_len
        self.out_dominance_freq = int(self.dominance_freq * self.length_ratio)

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(
                    nn.Linear(self.dominance_freq, self.out_dominance_freq).to(torch.cfloat)
                )
        else:
            self.freq_upsampler = nn.Linear(
                self.dominance_freq, self.out_dominance_freq
            ).to(torch.cfloat)

    def forward(self, x):
        low_specx = torch.fft.rfft(x, dim=-1)
        
        if self.individual:
            low_specxy_ = torch.zeros(
                [low_specx.size(0), self.channels, self.out_dominance_freq],
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
        
        return low_xy[:, :, -self.pred_len:]
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
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
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
            x = x / (self.affine_weight + self.eps*self.eps)
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
# 3. مكون التردد (Frequency Linear) - الإضافة الجديدة
# ==========================================
class FreqLinear(nn.Module):
    def __init__(self, seq_len, pred_len):
        super(FreqLinear, self).__init__()
        self.pred_len = pred_len
        
        # عند استخدام rfft، طول النطاق الترددي يكون النصف + 1
        freq_in_dim = seq_len // 2 + 1
        freq_out_dim = pred_len // 2 + 1
        
        # أوزان خطية منفصلة للجزء الحقيقي (Real) والتخيلي (Imaginary)
        self.linear_real = nn.Linear(freq_in_dim, freq_out_dim)
        self.linear_imag = nn.Linear(freq_in_dim, freq_out_dim)

    def forward(self, x):
        # x shape: [Batch, channels, seq_len]
        
        # 1. تحويل السلسلة إلى المجال الترددي
        x_fft = torch.fft.rfft(x, dim=-1)
        
        # 2. فصل الجزء الحقيقي والتخيلي
        real = x_fft.real
        imag = x_fft.imag
        
        # 3. تطبيق التحويل الخطي في المجال الترددي
        out_real = self.linear_real(real)
        out_imag = self.linear_imag(imag)
        
        # 4. إعادة دمج الأجزاء لتكوين أرقام مركبة
        out_fft = torch.complex(out_real, out_imag)
        
        # 5. التحويل العكسي للعودة إلى المجال الزمني بطول pred_len
        out = torch.fft.irfft(out_fft, n=self.pred_len, dim=-1)
        return out
# ==========================================
# 2. النموذج الهجين ثنائي المجال (Time-Frequency Dual-Domain DLinear)
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
        
        # 1. الاتجاه (المجال الزمني)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len, self.seq_len]))
        
        # 2. الموسمية في المجال الزمني (ممتازة لـ ETTh2 والأطوال الكبيرة)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len, self.seq_len]))
        
        # 3. الموسمية في المجال الترددي (ممتازة لـ ETTh1 والدورات المنتظمة)
        self.Freq_Seasonal = ComplexFreqLinear(self.seq_len, self.pred_len, self.enc_in, self.individual)
        
        # 4. بوابة دمج قابلة للتعلم لكل قناة (تبدأ متوازنة 0.5 عبر Sigmoid)
        self.alpha = nn.Parameter(torch.zeros(1, self.enc_in, 1))

    def forward(self, x):
        # x: [Batch, seq_len, channels]
        x = self.revin(x, 'norm')
        
        seasonal_init, trend_init = self.decomp(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        trend_output = self.Linear_Trend(trend_init)
        
        # حساب الموسمية في المجالين ودمجهما تكيفياً
        time_seasonal = self.Linear_Seasonal(seasonal_init)
        freq_seasonal = self.Freq_Seasonal(seasonal_init)
        
        w = torch.sigmoid(self.alpha) # قيمة محصورة بين 0 و 1 لكل قناة
        seasonal_output = w * time_seasonal + (1 - w) * freq_seasonal
        
        x_pred = (seasonal_output + trend_output).permute(0, 2, 1)
        x_pred = self.revin(x_pred, 'denorm')
        
        return x_pred
