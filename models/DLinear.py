import torch
import torch.nn as nn

# ==========================================
# 1. طبقة التقييس العكسي (RevIN)
# ==========================================
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
# 3. دالة المعادلة التفاضلية وكتلة RK4
# ==========================================
class ODEFunc(nn.Module):
    """
    تمثل المشتقة dh/dt = f_theta(h) في الفضاء الكامن.
    نستخدم دالة التفعيل Tanh لأنها ناعمة (Smooth) ومثالية للاستقرار العددي في RK4.
    """
    def __init__(self, d_model):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model)
        )
        # تهيئة الأوزان بقيم صغيرة في البداية لضمان استقرار التكامل العددي
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)

    def forward(self, h):
        return self.net(h)


class RK4Block(nn.Module):
    """
    كتلة التكامل العددي بطريقة رونج-كوتا من الرتبة الرابعة (Fixed-step RK4)
    """
    def __init__(self, d_model, num_steps=4):
        super(RK4Block, self).__init__()
        self.ode_func = ODEFunc(d_model)
        self.num_steps = num_steps
        self.dt = 1.0 / num_steps  # حجم الخطوة الزمنية

    def forward(self, h):
        for _ in range(self.num_steps):
            k1 = self.ode_func(h)
            k2 = self.ode_func(h + 0.5 * self.dt * k1)
            k3 = self.ode_func(h + 0.5 * self.dt * k2)
            k4 = self.ode_func(h + self.dt * k3)
            h = h + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return h


# ==========================================
# 4. النموذج المطور (RK4-DLinear + RevIN)
# ==========================================
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        enc_in = configs.enc_in
        kernel_size = 25

        # معاملات اختيارية للـ RK4 (تأخذ قيماً افتراضية إذا لم تكن موجودة في configs)
        d_model = getattr(configs, 'd_model', 128)
        rk4_steps = getattr(configs, 'rk4_steps', 8)

        # 1. طبقة التقييس RevIN
        self.revin = RevIN(num_features=enc_in)

        # 2. طبقة التفكيك
        self.decomp = series_decomp(kernel_size)

        # 3. مسار الموسمية (خطي كما في DLinear)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Seasonal.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
        )

        # 4. مسار الاتجاه الهجين (خطي + RK4 في الفضاء الكامن)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
        )

        self.trend_encoder = nn.Linear(self.seq_len, d_model)
        self.rk4_trend = RK4Block(d_model=d_model, num_steps=rk4_steps)
        self.trend_decoder = nn.Linear(d_model, self.pred_len)

        # معامل قابل للتعلم للتحكم في وزن مساهمة RK4 (يبدأ صغيراً للاستقرار)
        self.rk4_alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # x shape: [Batch, seq_len, channels]

        # أ. التقييس
        x = self.revin(x, 'norm')

        # ب. التفكيك
        seasonal_init, trend_init = self.decomp(x)

        # ج. تبديل الأبعاد إلى [Batch, channels, seq_len]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        # د. التنبؤ بالموسمية
        seasonal_output = self.Linear_Seasonal(seasonal_init)

        # هـ. التنبؤ بالاتجاه (دمج المسار الخطي مع المسار التفاضلي RK4)
        trend_linear = self.Linear_Trend(trend_init)
        
        trend_latent = self.trend_encoder(trend_init)
        trend_evolved = self.rk4_trend(trend_latent)
        trend_rk4 = self.trend_decoder(trend_evolved)

        trend_output = trend_linear + self.rk4_alpha * trend_rk4

        # و. دمج التنبؤات وإعادة الأبعاد
        x_pred = seasonal_output + trend_output
        x_pred = x_pred.permute(0, 2, 1)  # [Batch, pred_len, channels]

        # ز. التقييس العكسي
        x_pred = self.revin(x_pred, 'denorm')

        return x_pred
