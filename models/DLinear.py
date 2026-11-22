import torch
import torch.nn as nn


# ==========================================
# 1. مكون خلط القنوات (Channel Mixer)
# ==========================================
class ChannelMixer(nn.Module):
    """
    تطبيق شبكة MLP على بُعد القنوات (Channels) لالتقاط العلاقات بين المتغيرات المتعددة
    """
    def __init__(self, channels, dropout=0.1):
        super(ChannelMixer, self).__init__()
        # توسيع وتقليص الأبعاد لتعلم تمثيلات أعمق
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(), # تنشيط غير خطي لالتقاط الأنماط المعقدة
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout)
        )
        # تقييس الطبقة لتحقيق استقرار أثناء التدريب
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # الإدخال x: [Batch, pred_len, Channels]
        res = x
        x = self.norm(x)
        x = self.mlp(x)
        # اتصال متبقي (Residual Connection) لمنع تلاشي التدرجات
        return x + res
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
        dim2reduce = tuple(range(1, x.ndim - 1))
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
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


# ==========================================
# 2. مكون التفكيك (Series Decomposition)
# ==========================================
class moving_avg(nn.Module):
    """
    حساب المتوسط المتحرك لاستخراج مكون الاتجاه (Trend)
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Padding للحفاظ على طول السلسلة
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    تفكيك السلسلة الزمنية إلى: موسمية (Seasonal) واتجاه (Trend)
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# ==========================================
# 3. النموذج الكامل (DLinear + RevIN)
# ==========================================
# ==========================================
# 2. النموذج الكامل المدمج (DLinear + RevIN + Channel Mixer)
# ==========================================
class Model(nn.Module): 
    def __init__(self, configs): 
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        enc_in = configs.enc_in # عدد القنوات/المتغيرات
        kernel_size = 25 
        
        self.revin = RevIN(num_features=enc_in)
        self.decomp = series_decomp(kernel_size)
        
        # أ. مكون الخلط الزمني (Temporal Mixing - DLinear الأساسي)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        
        # ب. مكون خلط القنوات (Channel Mixing - الإضافة الجديدة)
        self.Seasonal_Channel_Mixer = ChannelMixer(enc_in, dropout=configs.dropout)
        self.Trend_Channel_Mixer = ChannelMixer(enc_in, dropout=configs.dropout)

    def forward(self, x):
        # x shape: [Batch, seq_len, channels]
        x = self.revin(x, 'norm')
        
        seasonal_init, trend_init = self.decomp(x)
        
        # تبديل الأبعاد لتطبيق الخلط الزمني
        seasonal_init = seasonal_init.permute(0, 2, 1) # [Batch, channels, seq_len]
        trend_init = trend_init.permute(0, 2, 1)       # [Batch, channels, seq_len]
        
        # 1. تطبيق الخلط الزمني
        seasonal_output = self.Linear_Seasonal(seasonal_init) # [Batch, channels, pred_len]
        trend_output = self.Linear_Trend(trend_init)          # [Batch, channels, pred_len]
        
        # إعادة الأبعاد لتطبيق خلط القنوات
        seasonal_output = seasonal_output.permute(0, 2, 1) # [Batch, pred_len, channels]
        trend_output = trend_output.permute(0, 2, 1)       # [Batch, pred_len, channels]
        
        # 2. تطبيق خلط القنوات
        seasonal_output = self.Seasonal_Channel_Mixer(seasonal_output)
        trend_output = self.Trend_Channel_Mixer(trend_output)
        
        x_pred = seasonal_output + trend_output
        
        x_pred = self.revin(x_pred, 'denorm')
        return x_pred
