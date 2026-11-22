import torch
import torch.nn as nn


# ==========================================
# مكون جديد: شبكة استخراج الأنماط الموسمية المعقدة
# ==========================================
class MLP_Seasonal(nn.Module):
    def __init__(self, seq_len, pred_len, d_model=128):
        super(MLP_Seasonal, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(), # دالة تنشيط غير خطية لالتقاط التغيرات الحادة والقمم
            nn.Dropout(0.1),
            nn.Linear(d_model, pred_len)
        )

    def forward(self, x):
        return self.mlp(x)

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
# 3. النموذج المطور (RevIN + Non-linear Seasonal + Linear Trend)
# ==========================================
class Model(nn.Module): 
    def __init__(self, configs): 
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        enc_in = configs.enc_in
        kernel_size = 25 
        
        # 1. طبقة التقييس
        self.revin = RevIN(num_features=enc_in)
        
        # 2. طبقة التفكيك
        self.decomp = series_decomp(kernel_size)
        
        # 3. طبقات التنبؤ الهجينة (Hybrid Forecasters)
        # استخدام MLP للموسمية لالتقاط الترددات العالية والقمم المعقدة
        self.Seasonal_Forecaster = MLP_Seasonal(self.seq_len, self.pred_len, d_model=128)
        
        # استخدام طبقة خطية للاتجاه لالتقاط النمو/الانحدار الرتيب بصلابة
        self.Trend_Forecaster = nn.Linear(self.seq_len, self.pred_len)
        
        # تهيئة أوزان طبقة الاتجاه
        self.Trend_Forecaster.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len, self.seq_len]))

    def forward(self, x):
        # أ. التقييس
        x = self.revin(x, 'norm')
        
        # ب. التفكيك
        seasonal_init, trend_init = self.decomp(x)
        
        # ج. تبديل الأبعاد: [Batch, channels, seq_len]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        # د. التنبؤ الهجين
        seasonal_output = self.Seasonal_Forecaster(seasonal_init)
        trend_output = self.Trend_Forecaster(trend_init)
        
        # هـ. الدمج
        x_pred = seasonal_output + trend_output
        x_pred = x_pred.permute(0, 2, 1) 
        
        # و. التقييس العكسي
        x_pred = self.revin(x_pred, 'denorm')
        
        return x_pred
