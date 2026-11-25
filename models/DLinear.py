import torch
import torch.nn as nn
import torch.fft


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


class multi_scale_decomp(nn.Module):
    """
    Multi-Scale Series Decomposition (Daily 25 + Weekly 169)
    """
    def __init__(self, kernel_sizes=[25, 169]):
        super(multi_scale_decomp, self).__init__()
        self.moving_avgs = nn.ModuleList([moving_avg(k, stride=1) for k in kernel_sizes])

    def forward(self, x):
        trends = [ma(x) for ma in self.moving_avgs]
        moving_mean = sum(trends) / len(trends)
        res = x - moving_mean
        return res, moving_mean


class ComplexFreqLinear(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, individual=False, dropout=0.1):
        super(ComplexFreqLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len
        self.individual = individual
        self.channels = enc_in
        
        self.dominance_freq = seq_len // 2 + 1
        self.length_ratio = self.total_len / self.seq_len
        self.dropout = nn.Dropout(p=dropout)

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(
                    nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(torch.cfloat)
                )
        else:
            self.freq_upsampler = nn.Linear(
                self.dominance_freq, int(self.dominance_freq * self.length_ratio)
            ).to(torch.cfloat)

    def forward(self, x):
        low_specx = torch.fft.rfft(x, dim=-1)
        
        if self.training:
            low_specx = torch.complex(self.dropout(low_specx.real), self.dropout(low_specx.imag))

        if self.individual:
            low_specxy_ = torch.zeros(
                [low_specx.size(0), self.channels, int(self.dominance_freq * self.length_ratio)],
                dtype=low_specx.dtype, device=low_specx.device
            )
            for i in range(self.channels):
                low_specxy_[:, i, :] = self.freq_upsampler[i](low_specx[:, i, :])
        else:
            low_specxy_ = self.freq_upsampler(low_specx)

        out_freq_dim = self.total_len // 2 + 1
        low_specxy = torch.zeros(
            [low_specxy_.size(0), low_specxy_.size(1), out_freq_dim],
            dtype=low_specxy_.dtype, device=low_specxy_.device
        )
        min_len = min(low_specxy_.size(-1), out_freq_dim)
        low_specxy[:, :, :min_len] = low_specxy_[:, :, :min_len]

        low_xy = torch.fft.irfft(low_specxy, n=self.total_len, dim=-1)
        low_xy = low_xy * self.length_ratio
        
        pred_seasonal = low_xy[:, :, -self.pred_len:]
        return pred_seasonal


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = getattr(configs, 'individual', False)

        self.revin = RevIN(num_features=self.channels)
        self.decomp = multi_scale_decomp(kernel_sizes=[25, 169])

        # ترك التهيئة الافتراضية ليتعلم الميل الحقيقي للاتجاه بدلاً من تجميده عند المتوسط
        if self.individual:
            self.Linear_Trend = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)
            ])
        else:
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        self.Freq_Seasonal = ComplexFreqLinear(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            enc_in=self.channels,
            individual=self.individual,
            dropout=0.1
        )

    def forward(self, x):
        x = self.revin(x, 'norm')

        seasonal_init, trend_init = self.decomp(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype, device=trend_init.device
            )
            for i in range(self.channels):
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            trend_output = self.Linear_Trend(trend_init)

        seasonal_output = self.Freq_Seasonal(seasonal_init)

        x_pred = (seasonal_output + trend_output).permute(0, 2, 1)
        x_pred = self.revin(x_pred, 'denorm')

        return x_pred
