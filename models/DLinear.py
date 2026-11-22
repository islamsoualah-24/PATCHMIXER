import torch
import torch.nn as nn

# ==========================================
# 1. طبقة الـ Patching
# ==========================================
class Patching(nn.Module):
    """
    تقسيم السلسلة الزمنية إلى أجزاء (Patches).
    باستخدام stride أقل من patch_len نحصل على تقطيع متداخل (Overlapping).
    """
    def __init__(self, patch_len, stride):
        super(Patching, self).__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x):
        # x shape: [Batch, Channels, Seq_len]
        # استخدام دالة unfold لاستخراج الأجزاء بفعالية
        # Output shape: [Batch, Channels, Num_patches, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return x

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
# 2. النموذج الكامل المقترح: Patch-DLinear
# ==========================================

class Model(nn.Module):
    def __init__(self, configs):
        super(PatchDLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # إعدادات الـ Patching (يمكنك إضافتها في ملف configs)
        self.patch_len = getattr(configs, 'patch_len', 16) # طول الجزء
        self.stride = getattr(configs, 'stride', 8)        # مقدار الانزياح (8 يعني تداخل بنسبة 50%)
        
        enc_in = configs.enc_in
        kernel_size = 25 
        
        self.revin = RevIN(num_features=enc_in)
        self.decomp = series_decomp(kernel_size)
        self.patching = Patching(self.patch_len, self.stride)
        
        # حساب عدد الـ Patches بناءً على طول السلسلة
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride) + 1
        
        # البعد الجديد بعد التقطيع والدمج
        self.patch_dim = self.num_patches * self.patch_len
        
        # الطبقات الخطية الآن تأخذ البعد الجديد (الذي يحتوي على الأنماط المحلية المتداخلة)
        self.Linear_Seasonal = nn.Linear(self.patch_dim, self.pred_len)
        self.Linear_Trend = nn.Linear(self.patch_dim, self.pred_len)
        
        # تهيئة الأوزان
        self.Linear_Seasonal.weight = nn.Parameter((1/self.patch_dim)*torch.ones([self.pred_len, self.patch_dim]))
        self.Linear_Trend.weight = nn.Parameter((1/self.patch_dim)*torch.ones([self.pred_len, self.patch_dim]))

    def forward(self, x):
        # x shape: [Batch, seq_len, channels]
        x = self.revin(x, 'norm')
        
        # التفكيك
        seasonal_init, trend_init = self.decomp(x)
        
        # تبديل الأبعاد لتجهيزها للتقطيع (التقطيع يتم على المحور الزمني)
        # Shape: [Batch, channels, seq_len]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        # تطبيق التقطيع (Patching)
        # Shape becomes: [Batch, channels, num_patches, patch_len]
        seasonal_patches = self.patching(seasonal_init)
        trend_patches = self.patching(trend_init)
        
        # تسطيح الأجزاء لتمريرها للطبقة الخطية مع الحفاظ على استقلالية القنوات
        # Shape becomes: [Batch, channels, num_patches * patch_len]
        seasonal_flat = seasonal_patches.reshape(seasonal_patches.size(0), seasonal_patches.size(1), -1)
        trend_flat = trend_patches.reshape(trend_patches.size(0), trend_patches.size(1), -1)
        
        # التنبؤ الخطي
        seasonal_output = self.Linear_Seasonal(seasonal_flat)
        trend_output = self.Linear_Trend(trend_flat)
        
        # دمج التنبؤات وإعادة الأبعاد
        x_pred = seasonal_output + trend_output
        x_pred = x_pred.permute(0, 2, 1) # Shape: [Batch, pred_len, channels]
        
        # التقييس العكسي
        x_pred = self.revin(x_pred, 'denorm')
        
        return x_pred
