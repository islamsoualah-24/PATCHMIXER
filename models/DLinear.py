import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. طبقة التقييس التكيفية (Adaptive RevIN)
# ==========================================
class AdaptiveRevIN(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        num_slices: int = 4,
        out_slices: int = 1,
        hidden_dim: int = 16,
        max_scale_shift: float = 0.5,
    ):
        super(AdaptiveRevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.num_slices = num_slices          # عدد شرائح الماضي لقراءة مسار التغير الإحصائي
        self.out_slices = out_slices          # عدد نقاط التحكم في المستقبل (1 لثابت، >1 لمسار متدرج)
        self.max_scale_shift = max_scale_shift # قيد أمان يمنع تضخم الانحراف المعياري

        if self.affine:
            self._init_params()

        # مشفّر خفيف مشترك بين القنوات (Channel-Independent) لمنع Overfitting
        self.shift_encoder = nn.Sequential(
            nn.Linear(2 * num_slices, hidden_dim),
            nn.GELU(),
        )

        # رأسان للتنبؤ بإزاحة المتوسط وتغير التباين (مهيآن بالأصفار لضمان استقرار خط الأساس)
        self.mean_shift_head = nn.Linear(hidden_dim, out_slices)
        self.std_shift_head = nn.Linear(hidden_dim, out_slices)
        nn.init.zeros_(self.mean_shift_head.weight)
        nn.init.zeros_(self.mean_shift_head.bias)
        nn.init.zeros_(self.std_shift_head.weight)
        nn.init.zeros_(self.std_shift_head.bias)

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _predict_future_shifts(self, x_norm):
        """
        استخراج مسار المتوسط والانحراف المعياري عبر شرائح الماضي المُقيّس
        والتنبؤ بالإزاحة الإحصائية المستقبلية.
        x_norm shape: [Batch, seq_len, channels]
        """
        x_t = x_norm.permute(0, 2, 1)  # [Batch, channels, seq_len]

        # حساب متوسط وانحراف كل شريحة (يعمل مع أي قيمة لـ seq_len)
        seg_mean = F.adaptive_avg_pool1d(x_t, self.num_slices)
        seg_sq_mean = F.adaptive_avg_pool1d(x_t ** 2, self.num_slices)
        seg_var = torch.clamp(seg_sq_mean - seg_mean ** 2, min=0.0)
        seg_std = torch.sqrt(seg_var + self.eps)

        # دمج السمات الإحصائية: [Batch, channels, 2 * num_slices]
        stats = torch.cat([seg_mean, seg_std], dim=-1)
        h = self.shift_encoder(stats)

        # حفظ إزاحات المستقبل لاستخدامها عند denorm
        self.delta_mean_slices = self.mean_shift_head(h)  # [Batch, channels, out_slices]
        self.delta_log_std_slices = self.max_scale_shift * torch.tanh(
            self.std_shift_head(h)
        )

    def _normalize(self, x):
        x_norm = (x - self.mean) / self.stdev
        # التنبؤ بالإزاحة المستقبلية قبل تطبيق affine
        self._predict_future_shifts(x_norm)

        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
        return x_norm

    def _denormalize(self, x):
        # x shape: [Batch, pred_len, channels]
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)

        pred_len = x.size(1)

        # تمديد نقاط الإزاحة بسلاسة لتغطي كامل أفق التنبؤ pred_len
        if self.out_slices > 1 and pred_len > 1:
            delta_mean = F.interpolate(
                self.delta_mean_slices, size=pred_len, mode='linear', align_corners=True
            ).permute(0, 2, 1)
            delta_log_std = F.interpolate(
                self.delta_log_std_slices, size=pred_len, mode='linear', align_corners=True
            ).permute(0, 2, 1)
        else:
            delta_mean = self.delta_mean_slices.mean(dim=-1, keepdim=True).permute(0, 2, 1)
            delta_log_std = self.delta_log_std_slices.mean(dim=-1, keepdim=True).permute(0, 2, 1)

        # حساب المتوسط والانحراف المعياري المستقبليين المتكيفين
        future_std = self.stdev * torch.exp(delta_log_std)
        future_mean = self.mean + delta_mean * self.stdev

        x = x * future_std + future_mean
        return x

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x


# ==========================================
# 2. مكون التفكيك (Series Decomposition) - دون تغيير
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
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# ==========================================
# 3. النموذج الكامل (DLinear + Adaptive RevIN)
# ==========================================
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        enc_in = configs.enc_in
        kernel_size = 25

        # معاملات اختيارية من configs مع قيم افتراضية آمنة
        num_slices = getattr(configs, 'rev_num_slices', 4)
        out_slices = getattr(configs, 'rev_out_slices', 4)

        # 1. طبقة التقييس التكيفية Adaptive RevIN
        self.revin = AdaptiveRevIN(
            num_features=enc_in,
            num_slices=num_slices,
            out_slices=out_slices,
        )

        # 2. طبقة التفكيك
        self.decomp = series_decomp(kernel_size)

        # 3. الطبقات الخطية للتنبؤ (بنفس التهيئة الناجحة تماماً)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        self.Linear_Seasonal.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
        )
        self.Linear_Trend.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
        )

    def forward(self, x):
        # x shape: [Batch, seq_len, channels]
        x = self.revin(x, 'norm')

        seasonal_init, trend_init = self.decomp(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x_pred = (seasonal_output + trend_output).permute(0, 2, 1)
        x_pred = self.revin(x_pred, 'denorm')

        return x_pred
