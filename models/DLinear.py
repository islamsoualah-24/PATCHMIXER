import torch
import torch.nn as nn
import torch.fft

# ==========================================
# 1. Reversible Instance Normalization (RevIN)
# ==========================================
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_affine_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_affine_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

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
# 2. Series Decomposition (Trend & Seasonal)
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
# 3. Fourier Seasonal Forecaster (الجديد)
# ==========================================
class FourierSeasonal(nn.Module):
    def __init__(self, seq_len, pred_len, channels):
        super(FourierSeasonal, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # حساب عدد الترددات المدخلة والمخرجة بناءً على تحويل فورييه السريع للأعداد الحقيقية (rfft)
        self.in_freqs = seq_len // 2 + 1
        self.out_freqs = pred_len // 2 + 1
        
        # مصفوفة أوزان قابلة للتعلم في المجال الترددي (بأرقام مركبة Complex Numbers)
        # هذا يسمح للنموذج بفلترة وتضخيم الترددات المهمة لكل قناة
        self.complex_weight = nn.Parameter(
            torch.randn(channels, self.out_freqs, self.in_freqs, dtype=torch.cfloat) * 0.02
        )

    def forward(self, x):
        # x shape: [Batch, Channels, Seq_len]
        
        # 1. التحويل للمجال الترددي
        x_fft = torch.fft.rfft(x, dim=-1) # Output shape: [Batch, Channels, in_freqs]
        
        # 2. تطبيق الفلتر الترددي الخطي (ضرب مصفوفات باستخدام einsum)
        # b: Batch, c: Channels, i: in_freqs, o: out_freqs
        out_fft = torch.einsum('bci,coi->bco', x_fft, self.complex_weight)
        
        # 3. التحويل العكسي للمجال الزمني بحجم التنبؤ المطلوب
        x_pred = torch.fft.irfft(out_fft, n=self.pred_len, dim=-1) # Output shape: [Batch, Channels, Pred_len]
        return x_pred

# ==========================================
# 4. النموذج الكامل (Fourier-enhanced DLinear)
# ==========================================
class Model(nn.Module): 
    def __init__(self, configs): 
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        kernel_size = 25 
        
        # التقييس والتفكيك
        self.revin = RevIN(num_features=self.channels)
        self.decomp = series_decomp(kernel_size)
        
        # طبقة التنبؤ الموسمي (عبر فورييه)
        self.Seasonal_Forecaster = FourierSeasonal(self.seq_len, self.pred_len, self.channels)
        
        # طبقة التنبؤ للاتجاه (طبقة خطية كلاسيكية)
        self.Trend_Forecaster = nn.Linear(self.seq_len, self.pred_len)
        self.Trend_Forecaster.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len, self.seq_len]))

    def forward(self, x):
        # x shape: [Batch, Seq_len, Channels]
        
        x = self.revin(x, 'norm')
        seasonal_init, trend_init = self.decomp(x)
        
        seasonal_init = seasonal_init.permute(0, 2, 1) # [Batch, Channels, Seq_len]
        trend_init = trend_init.permute(0, 2, 1)       # [Batch, Channels, Seq_len]
        
        seasonal_output = self.Seasonal_Forecaster(seasonal_init) # [Batch, Channels, Pred_len]
        trend_output = self.Trend_Forecaster(trend_init)          # [Batch, Channels, Pred_len]
        
        x_pred = seasonal_output + trend_output
        x_pred = x_pred.permute(0, 2, 1) # [Batch, Pred_len, Channels]
        
        x_pred = self.revin(x_pred, 'denorm')
        
        return x_pred
