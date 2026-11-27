import torch
import torch.nn as nn
import torch.fft


# ==========================================
# 1. مكون التردد المركب (Complex Frequency Upsampling)
# ==========================================
class ComplexFreqLinear(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, individual=False):
        super(ComplexFreqLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len
        self.individual = individual
        self.channels = enc_in

        self.dominance_freq = seq_len // 2 + 1
        self.length_ratio = self.total_len / self.seq_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for _ in range(self.channels):
                self.freq_upsampler.append(
                    nn.Linear(
                        self.dominance_freq,
                        int(self.dominance_freq * self.length_ratio)
                    ).to(torch.cfloat)
                )
        else:
            self.freq_upsampler = nn.Linear(
                self.dominance_freq,
                int(self.dominance_freq * self.length_ratio)
            ).to(torch.cfloat)

    def forward(self, x):
        # x shape: [Batch, channels, seq_len]
        low_specx = torch.fft.rfft(x, dim=-1)

        if self.individual:
            low_specxy_ = torch.zeros(
                [low_specx.size(0), self.channels, int(self.dominance_freq * self.length_ratio)],
                dtype=low_specx.dtype,
                device=low_specx.device
            )
            for i in range(self.channels):
                low_specxy_[:, i, :] = self.freq_upsampler[i](low_specx[:, i, :])
        else:
            low_specxy_ = self.freq_upsampler(low_specx)

        out_freq_dim = self.total_len // 2 + 1
        low_specxy = torch.zeros(
            [low_specxy_.size(0), low_specxy_.size(1), out_freq_dim],
            dtype=low_specxy_.dtype,
            device=low_specxy_.device
        )
        min_len = min(low_specxy_.size(-1), out_freq_dim)
        low_specxy[:, :, :min_len] = low_specxy_[:, :, :min_len]

        low_xy = torch.fft.irfft(low_specxy, n=self.total_len, dim=-1)
        low_xy = low_xy * self.length_ratio

        pred_seasonal = low_xy[:, :, -self.pred_len:]
        return pred_seasonal


# ==========================================
# 2. طبقة التقييس (RevIN)
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
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()

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
# 3. مكون التفكيك (Series Decomposition)
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
# 4. كتلة التصحيح الديناميكي المتبقي (RK4 Residual Correction) - الإضافة الجديدة
# ==========================================
class RK4ResidualCorrection(nn.Module):
    """
    تعمل كتصحيح ديناميكي غير خطي فوق مخرج Linear_Trend:
    T_out = T_linear + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    """
    def __init__(self, pred_len, hidden_dim=32, num_steps=1, max_h=1.0):
        super(RK4ResidualCorrection, self).__init__()
        self.num_steps = num_steps
        self.max_h = max_h

        # شبكة ديناميكية صغيرة ومقيدة (Bounded & Lipschitz-constrained)
        self.f_theta = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(nn.Linear(pred_len, hidden_dim)),
            nn.Tanh(),
            nn.utils.parametrizations.spectral_norm(nn.Linear(hidden_dim, pred_len)),
            nn.Tanh()
        )

        # حجم خطوة h قابل للتعلم ومقيد ضمن (0, max_h)
        # التهيئة عند -3.0 تعطي h ≈ 0.047 في بداية التدريب لضمان استقرار المسار الخطي
        self.raw_h = nn.Parameter(torch.tensor(-3.0))

    def forward(self, z_linear):
        # z_linear shape: [Batch, channels, pred_len]
        h = (self.max_h * torch.sigmoid(self.raw_h)) / self.num_steps
        z = z_linear

        for _ in range(self.num_steps):
            k1 = self.f_theta(z)
            k2 = self.f_theta(z + 0.5 * h * k1)
            k3 = self.f_theta(z + 0.5 * h * k2)
            k4 = self.f_theta(z + h * k3)
            z = z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return z


# ==========================================
# 5. النموذج المدمج (Complex-Freq DLinear + RK4 Residual Trend + RevIN)
# ==========================================
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.individual = getattr(configs, 'individual', False)
        rk4_hidden = getattr(configs, 'rk4_hidden', 32)
        rk4_steps = getattr(configs, 'rk4_steps', 1)
        kernel_size = 25

        self.revin = RevIN(num_features=self.enc_in)
        self.decomp = series_decomp(kernel_size)

        # 1) المسار العام للاتجاه (الجزء الخطي الأساسي كما هو)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
        )

        # 2) التصحيح الديناميكي المتبقي فوق الاتجاه الخطي (RK4)
        self.RK4_Trend_Correction = RK4ResidualCorrection(
            pred_len=self.pred_len,
            hidden_dim=rk4_hidden,
            num_steps=rk4_steps
        )

        # 3) الموسمية في المجال الترددي المركب
        self.Freq_Seasonal = ComplexFreqLinear(
            self.seq_len, self.pred_len, self.enc_in, self.individual
        )

    def forward(self, x):
        # x: [Batch, seq_len, channels]
        x = self.revin(x, 'norm')

        seasonal_init, trend_init = self.decomp(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        # المسار الخطي العام + التصحيح الديناميكي غير الخطي عبر RK4
        trend_linear = self.Linear_Trend(trend_init)
        trend_output = self.RK4_Trend_Correction(trend_linear)

        seasonal_output = self.Freq_Seasonal(seasonal_init)

        x_pred = (seasonal_output + trend_output).permute(0, 2, 1)
        x_pred = self.revin(x_pred, 'denorm')

        return x_pred
