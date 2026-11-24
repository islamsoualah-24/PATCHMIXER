import torch
import torch.nn as nn
import torch.fft # ضروري لتحويلات فورييه

# ==========================================
# 1. مكون التردد الصحيح (Complex Frequency Upsampling)
# ==========================================
class ComplexFreqLinear(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, individual=False, cut_freq=25):
        super(ComplexFreqLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len
        self.individual = individual
        self.channels = enc_in
        
        # تطبيق Low-Pass Filter: الاحتفاظ فقط بأول cut_freq تردد وحذف الضوضاء العالية
        self.dominance_freq = min(cut_freq, seq_len // 2 + 1)
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
        # x shape: [Batch, channels, seq_len]
        
        # 1. تحويل فورييه السريع واقتطاع الترددات العالية (Low-Pass Filtering)
        low_specx = torch.fft.rfft(x, dim=-1)[:, :, :self.dominance_freq]
        
        # 2. الإسقاط الخطي المركب لتوسيع الطيف الترددي
        if self.individual:
            low_specxy_ = torch.zeros(
                [low_specx.size(0), self.channels, self.out_dominance_freq],
                dtype=low_specx.dtype, device=low_specx.device
            )
            for i in range(self.channels):
                low_specxy_[:, i, :] = self.freq_upsampler[i](low_specx[:, i, :])
        else:
            low_specxy_ = self.freq_upsampler(low_specx)

        # 3. حشو باقي الطيف بالأصفار (Zero-Padding) ليطابق الطول الكلي (seq_len + pred_len)
        out_freq_dim = self.total_len // 2 + 1
        low_specxy = torch.zeros(
            [low_specxy_.size(0), low_specxy_.size(1), out_freq_dim],
            dtype=low_specxy_.dtype, device=low_specxy_.device
        )
        min_len = min(low_specxy_.size(-1), out_freq_dim)
        low_specxy[:, :, :min_len] = low_specxy_[:, :, :min_len]

        # 4. التحويل العكسي وضرب الإشارة في معامل التصحيح للحفاظ على الطاقة
        low_xy = torch.fft.irfft(low_specxy, n=self.total_len, dim=-1)
        low_xy = low_xy * self.length_ratio
        
        # 5. استخراج الجزء الخاص بالتنبؤ فقط (آخر pred_len خطوة)
        pred_seasonal = low_xy[:, :, -self.pred_len:]
        return pred_seasonal
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
# 2. النموذج المدمج (Complex-Freq DLinear + RevIN)
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
        
        # الاتجاه في المجال الزمني
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len, self.seq_len]))
        
        # الموسمية في المجال الترددي المركب
        self.Freq_Seasonal = ComplexFreqLinear(self.seq_len, self.pred_len, self.enc_in, self.individual)

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
