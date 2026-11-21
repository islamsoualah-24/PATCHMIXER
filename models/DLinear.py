import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear with Channel Mixing (Cross-Channel Dependencies)
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual

        # 1. Series Decomposition
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)

        # 2. Temporal Linear Layers (Independent or Shared)
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        # ---------------------------------------------------------
        # التعديل الجديد: إضافة آلية خلط القنوات (Channel Mixing)
        # ---------------------------------------------------------
        # نستخدم شبكة MLP صغيرة تعمل على بعد القنوات لتبادل المعلومات بينها
        # نستخدم عامل توسيع (Expansion Factor) بسيط لتجنب زيادة المعلمات بشكل كبير
        expansion_factor = 2
        self.channel_mixer = nn.Sequential(
            nn.Linear(self.channels, self.channels * expansion_factor),
            nn.GELU(),
            nn.Dropout(0.1), # Dropout خفيف لمنع التكيف الزائد
            nn.Linear(self.channels * expansion_factor, self.channels)
        )
        
        # طبقة تسوية (Layer Normalization) لضمان استقرار التدريب قبل الخلط
        self.norm = nn.LayerNorm(self.channels)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        
        # 1. Decomposition
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        
        # 2. Temporal Processing
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len], dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len], dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        # دمج المكونات الزمنية
        x_temporal = seasonal_output + trend_output # Shape: [Batch, Channel, Pred_len]
        
        # إعادة التشكيل لتصبح القنوات هي البعد الأخير لتطبيق الطبقات الخطية عليها
        x_temporal = x_temporal.permute(0, 2, 1) # Shape: [Batch, Pred_len, Channel]

        # ---------------------------------------------------------
        # تطبيق خلط القنوات (Channel Mixing)
        # ---------------------------------------------------------
        # تطبيق التسوية أولاً (Pre-norm) ثم شبكة الخلط، مع اتصال التفافي (Residual Connection)
        x_mixed = self.channel_mixer(self.norm(x_temporal))
        out = x_temporal + x_mixed
        
        return out # Shape: [Batch, Pred_len, Channel]
