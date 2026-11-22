import torch
import torch.nn as nn


class AttentionalSeasonal(nn.Module):
    """
    مكون موسمي يعتمد على آلية الانتباه الذاتي لالتقاط التموجات المعقدة والقمم.
    """
    def __init__(self, seq_len, pred_len, channels, num_heads=4):
        super(AttentionalSeasonal, self).__init__()
        # طبقة الانتباه الذاتي متعدد الرؤوس (تعمل عبر القنوات)
        self.self_attention = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        
        # طبقة تسوية (Normalization) لضمان استقرار التدريب
        self.norm = nn.LayerNorm(channels)
        
        # طبقة خطية نهائية لتحويل طول السلسلة المدخلة إلى طول التنبؤ المطلوب
        self.linear_projection = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x shape: [Batch, Seq_len, Channels]
        
        # 1. تطبيق الانتباه الذاتي
        attn_out, _ = self.self_attention(query=x, key=x, value=x)
        
        # 2. اتصال متبقي (Residual Connection) + تسوية
        x = self.norm(x + attn_out)
        
        # 3. تبديل الأبعاد لتطبيق التحويل الخطي على المحور الزمني
        # Shape becomes: [Batch, Channels, Seq_len]
        x = x.permute(0, 2, 1)
        
        # 4. التنبؤ الخطي للطول المستهدف
        # Shape becomes: [Batch, Channels, Pred_len]
        x = self.linear_projection(x)
        
        # إعادة الأبعاد لحالتها الأصلية: [Batch, Pred_len, Channels]
        return x.permute(0, 2, 1)

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
class Model(nn.Module):
    """
    النموذج الهجين: الانتباه الذاتي للموسمية + خطي للاتجاه
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        kernel_size = 25 
        
        # التقييس والتفكيك
        self.revin = RevIN(num_features=self.channels)
        self.decomp = series_decomp(kernel_size)
        
        # استخدام المكون الجديد المبني على الانتباه للموسمية
        # يجب أن يكون عدد القنوات (channels) قابلاً للقسمة على num_heads
        # إذا كان enc_in مثلاً 7، يمكنك استخدام num_heads=1 أو تعديل الأبعاد
        self.Seasonal_Block = AttentionalSeasonal(self.seq_len, self.pred_len, self.channels, num_heads=1)
        
        # إبقاء الاتجاه خطياً
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len, self.seq_len]))

    def forward(self, x):
        # x shape: [Batch, seq_len, channels]
        x = self.revin(x, 'norm')
        
        # التفكيك
        seasonal_init, trend_init = self.decomp(x)
        
        # المكون الموسمي (الانتباه الذاتي يعمل مباشرة على [Batch, seq_len, channels])
        seasonal_output = self.Seasonal_Block(seasonal_init)
        
        # المكون الاتجاهي (نحتاج لتبديل الأبعاد لتطبيق الطبقة الخطية على الزمن)
        trend_init = trend_init.permute(0, 2, 1)
        trend_output = self.Linear_Trend(trend_init)
        trend_output = trend_output.permute(0, 2, 1)
        
        # دمج التنبؤات
        x_pred = seasonal_output + trend_output
        
        # التقييس العكسي
        x_pred = self.revin(x_pred, 'denorm')
        
        return x_pred
