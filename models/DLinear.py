import torch
import torch.nn as nn
import torch.fft # ضروري لتحويلات فورييه

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
# 4. النموذج الكامل (F-DLinear + RevIN)
# ==========================================
class Model(nn.Module): 
    def __init__(self, configs): 
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        enc_in = configs.enc_in
        kernel_size = 25 
        
        # 1. طبقة التقييس RevIN
        self.revin = RevIN(num_features=enc_in)
        
        # 2. طبقة التفكيك
        self.decomp = series_decomp(kernel_size)
        
        # 3. الطبقات الخطية (هنا يكمن التغيير الجوهري)
        # الاتجاه (Trend) يبقى في المجال الزمني لأنه سلس ورتيب
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len, self.seq_len]))
        
        # الموسمية (Seasonal) تنتقل للمجال الترددي لالتقاط القمم والتذبذبات
        self.Freq_Seasonal = FreqLinear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x shape: [Batch, seq_len, channels]
        
        # أ. التقييس
        x = self.revin(x, 'norm')
        
        # ب. التفكيك
        seasonal_init, trend_init = self.decomp(x)
        
        # ج. تبديل الأبعاد
        seasonal_init = seasonal_init.permute(0, 2, 1) # [Batch, channels, seq_len]
        trend_init = trend_init.permute(0, 2, 1)       # [Batch, channels, seq_len]
        
        # د. التنبؤ (هجين: زمني للاتجاه، وترددي للموسمية)
        trend_output = self.Linear_Trend(trend_init)
        seasonal_output = self.Freq_Seasonal(seasonal_init)
        
        # هـ. دمج التنبؤات
        x_pred = seasonal_output + trend_output
        x_pred = x_pred.permute(0, 2, 1) # [Batch, pred_len, channels]
        
        # و. التقييس العكسي
        x_pred = self.revin(x_pred, 'denorm')
        
        return x_pred
