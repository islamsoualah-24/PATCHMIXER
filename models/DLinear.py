import torch
import torch.nn as nn
import torch.fft

# ==========================================
# 1. طبقة التقييس (RevIN) - مطابقة لنسختك الفائزة
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
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev + self.mean
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
# 3. خليط الخبراء الخطيين الموجه ترددياً (Spectral-Guided MoLE)
# ==========================================
class SpectralMoLE(nn.Module):
    def __init__(self, seq_len, pred_len, num_experts=10, max_freq_bins=32, dropout=0.1):
        super(SpectralMoLE, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_experts = num_experts
        
        # الاكتفاء بأهم الترددات المنخفضة لتجنب الضوضاء وتقليل معاملات الموجّه
        self.freq_bins = min(seq_len // 2 + 1, max_freq_bins)
        router_in_dim = self.freq_bins * 2  # الجزء الحقيقي + الجزء التخيلي (السعة والطور معاً)
        router_hidden = max(router_in_dim // 2, 16)

        # الموجّه الطيفي (Spectral Router)
        self.router_norm = nn.LayerNorm(router_in_dim)
        self.router = nn.Sequential(
            nn.Linear(router_in_dim, router_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden, num_experts)
        )
        # تصفير الطبقة الأخيرة ليبدأ التوزيع متساوياً تماماً (1/K) لكل خبير
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

        # جميع الخبراء في طبقة خطية مدمجة واحدة للسرعة القصوى
        self.experts = nn.Linear(seq_len, pred_len * num_experts)
        self._init_experts()

        # لحفظ أوزان التوجيه لتحليلها وتفسيرها في فصل النتائج بالأطروحة
        self.last_routing_weights = None

    def _init_experts(self):
        with torch.no_grad():
            base_w = (1.0 / self.seq_len) * torch.ones(self.num_experts, self.pred_len, self.seq_len)
            # ضوضاء دقيقة ذات متوسط صفري عبر الخبراء: تكسر التماثل وتحافظ على نقطة بداية DLinear بدقة
            noise = torch.randn_like(base_w) * (1e-2 / self.seq_len)
            noise = noise - noise.mean(dim=0, keepdim=True)
            
            self.experts.weight.copy_((base_w + noise).view(self.num_experts * self.pred_len, self.seq_len))
            self.experts.bias.zero_()

    def forward(self, x):
        # x shape: [Batch, channels, seq_len]
        B, C, _ = x.shape

        # 1. استخراج البصمة الترددية الواعية بالطور
        x_fft = torch.fft.rfft(x, dim=-1)[:, :, :self.freq_bins]
        spec_feat = torch.cat([x_fft.real, x_fft.imag], dim=-1)  # [B, C, 2 * freq_bins]
        spec_feat = self.router_norm(spec_feat)

        # 2. حساب أوزان اختيار الخبراء لكل نافذة ولكل قناة
        logits = self.router(spec_feat)          # [B, C, num_experts]
        weights = torch.softmax(logits, dim=-1)  # [B, C, num_experts]
        self.last_routing_weights = weights.detach()

        # 3. حساب تنبؤات جميع الخبراء ودمجها حسب أوزان الموجّه
        all_preds = self.experts(x).view(B, C, self.num_experts, self.pred_len)
        out = torch.sum(all_preds * weights.unsqueeze(-1), dim=2)  # [B, C, pred_len]
        return out


# ==========================================
# 4. النموذج الكامل (Spectral-MoLE DLinear + RevIN)
# ==========================================
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        enc_in = configs.enc_in
        kernel_size = 25
        
        # يمكنك تمرير num_experts في configs أو تركه على القيمة الافتراضية 4
        num_experts = getattr(configs, 'num_experts', 4)
        dropout = getattr(configs, 'dropout', 0.1)

        self.revin = RevIN(num_features=enc_in)
        self.decomp = series_decomp(kernel_size)

        # استبدال الطبقتين الخطيتين بخليط الخبراء الموجه ترددياً
        self.MoLE_Seasonal = SpectralMoLE(self.seq_len, self.pred_len, num_experts=num_experts, dropout=dropout)
        self.MoLE_Trend = SpectralMoLE(self.seq_len, self.pred_len, num_experts=num_experts, dropout=dropout)

    def forward(self, x):
        # x shape: [Batch, seq_len, channels]
        x = self.revin(x, 'norm')

        seasonal_init, trend_init = self.decomp(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        seasonal_output = self.MoLE_Seasonal(seasonal_init)
        trend_output = self.MoLE_Trend(trend_init)

        x_pred = (seasonal_output + trend_output).permute(0, 2, 1)
        x_pred = self.revin(x_pred, 'denorm')

        return x_pred
