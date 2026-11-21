import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    (بقي كما هو دون تغيير)
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


class MultiScale_SeriesDecomp(nn.Module):
    """
    Multi-Scale Series decomposition block
    آلية التفكيك الجديدة متعددة المقاييس لأطروحة الدكتوراه
    """
    def __init__(self, kernel_sizes):
        super(MultiScale_SeriesDecomp, self).__init__()
        # إنشاء قائمة من طبقات المتوسط المتحرك بناءً على النوافذ المختلفة
        self.moving_avg_layers = nn.ModuleList([
            moving_avg(kernel_size, stride=1) for kernel_size in kernel_sizes
        ])

    def forward(self, x):
        moving_means = []
        # تمرير السلسلة الزمنية على كل نافذة مقياس
        for moving_avg_layer in self.moving_avg_layers:
            moving_means.append(moving_avg_layer(x))
        
        # دمج الاتجاهات المختلفة (هنا نستخدم المتوسط كبداية)
        # sum(moving_means) يجمع التنسورات، والقسمة للحصول على المتوسط
        trend = sum(moving_means) / len(moving_means)
        
        # استخراج الجزء الموسمي
        res = x - trend
        return res, trend


class Model(nn.Module):
    """
    Non-Linear Multi-Scale Decomposition (NL-MS-DLinear)
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # بُعد مخفي للشبكة غير الخطية (يمكنك تعديله، 64 أو 128 عادة ما تعطي نتائج جيدة)
        d_hidden = 128 
        dropout = 0.1

        kernel_sizes = [7, 15, 25, 51] 
        self.decompsition = MultiScale_SeriesDecomp(kernel_sizes)
        
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Trend = nn.ModuleList()
            self.MLP_Seasonal = nn.ModuleList() # تغيير الاسم ليناسب المعمارية الجديدة
            
            for i in range(self.channels):
                # الاتجاه يبقى خطياً
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
                # الجزء الموسمي يصبح غير خطي لالتقاط القمم الحادة
                self.MLP_Seasonal.append(
                    nn.Sequential(
                        nn.Linear(self.seq_len, d_hidden),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(d_hidden, self.pred_len)
                    )
                )
        else:
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.MLP_Seasonal = nn.Sequential(
                nn.Linear(self.seq_len, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, self.pred_len)
            )

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len], dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len], dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.MLP_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.MLP_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1) # to [Batch, Output length, Channel]
