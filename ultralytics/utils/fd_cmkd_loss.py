import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("FDFeatureAdapter", "FD_NLKD_Loss")


class FDFeatureAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.proj = nn.Identity() if self.in_channels == self.out_channels else nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class FDAttentionMatrix(nn.Module):
    def __init__(self, in_channels, inter_channels, sub_sample=True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.inter_channels = int(inter_channels)
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1)
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1)
        if sub_sample:
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, y_s, y_t):
        if y_s.shape != y_t.shape:
            raise RuntimeError(f"FD-CMKD feature shape mismatch: {tuple(y_s.shape)} vs {tuple(y_t.shape)}")
        batch_size = y_s.size(0)
        theta_y_s = self.theta(y_s).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        phi_y_t = self.phi(y_t).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_y_s, phi_y_t)
        return f / f.size(-1)


class FDNonLocalAttention(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.inter_channels = int(inter_channels) if inter_channels is not None else max(self.in_channels // 2, 1)
        self.att = FDAttentionMatrix(self.in_channels, self.inter_channels, sub_sample=sub_sample)
        self.g = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1)
        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(self.in_channels),
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1, stride=1)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, y_s, y_t):
        if y_s.shape != y_t.shape:
            raise RuntimeError(f"FD-CMKD feature shape mismatch: {tuple(y_s.shape)} vs {tuple(y_t.shape)}")
        batch_size = y_s.size(0)
        f_div_c = self.att(y_s, y_t)
        g_y_t = self.g(y_t).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        y_s_ = torch.matmul(f_div_c, g_y_t)
        y_s_ = y_s_.permute(0, 2, 1).contiguous().view(batch_size, self.inter_channels, *y_s.size()[2:])
        w_y_s = self.W(y_s_)
        z_s = w_y_s + y_s
        z_t = y_t
        return z_s, z_t


class DCFilter2d(nn.Module):
    def forward(self, x):
        return x - x.mean(dim=(-2, -1), keepdim=True)


class LowHighPassFilter2d(nn.Module):
    def __init__(self, low_keep_ratio=0.25):
        super().__init__()
        self.low_keep_ratio = float(low_keep_ratio)

    def _low_mask(self, height, width, device, dtype):
        width_rfft = width // 2 + 1
        keep_ratio = min(max(self.low_keep_ratio, 1e-3), 1.0)
        keep_h = max(1, min(height, int(round(height * keep_ratio))))
        keep_w = max(1, min(width_rfft, int(round(width_rfft * keep_ratio))))
        mask = torch.zeros((1, 1, height, width_rfft), device=device, dtype=dtype)
        mask[..., :keep_h, :keep_w] = 1
        return mask

    def forward(self, x):
        if x.ndim != 4:
            raise RuntimeError(f"FD-CMKD expects 4D feature maps, got shape {tuple(x.shape)}")
        height, width = x.shape[-2:]
        freq = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        low_mask = self._low_mask(height, width, device=x.device, dtype=freq.real.dtype)
        high_mask = 1.0 - low_mask
        low = torch.fft.irfft2(freq * low_mask, s=(height, width), dim=(-2, -1), norm="ortho").real
        high = torch.fft.irfft2(freq * high_mask, s=(height, width), dim=(-2, -1), norm="ortho").real
        return low, high


def signed_log1p(x):
    pos = torch.log1p(torch.clamp_min(x, 0))
    neg = -torch.log1p(torch.clamp_min(-x, 0))
    return torch.where(x >= 0, pos, neg)


def l2_normalize_feature(x, eps=1e-8):
    norm = x.flatten(1).norm(p=2, dim=1, keepdim=True).clamp_min(eps)
    return x / norm.view(-1, 1, 1, 1)


class FD_NLKD_Loss(nn.Module):
    def __init__(
        self,
        in_channels,
        inter_channels=None,
        sub_sample=True,
        bn_layer=True,
        low_keep_ratio=0.25,
        low_freq_weight=1.0,
        high_freq_weight=1.0,
        loss_weight=1.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.inter_channels = int(inter_channels) if inter_channels is not None else max(self.in_channels // 2, 1)
        self.low_keep_ratio = float(low_keep_ratio)
        self.low_freq_weight = float(low_freq_weight)
        self.high_freq_weight = float(high_freq_weight)
        self.loss_weight = float(loss_weight)
        self.non_local_att = FDNonLocalAttention(
            self.in_channels,
            self.inter_channels,
            sub_sample=sub_sample,
            bn_layer=bn_layer,
        )
        self.dc_filter = DCFilter2d()
        self.fd_filter = LowHighPassFilter2d(low_keep_ratio=self.low_keep_ratio)
        self.last_low_loss = None
        self.last_high_loss = None

    def _standardize(self, feat):
        feat = self.dc_filter(feat)
        return l2_normalize_feature(feat)

    def forward(self, s_feature, t_feature):
        if s_feature.shape != t_feature.shape:
            raise RuntimeError(f"FD-CMKD feature shape mismatch: {tuple(s_feature.shape)} vs {tuple(t_feature.shape)}")
        z_s, z_t = self.non_local_att(s_feature, t_feature)
        z_s = self._standardize(z_s)
        z_t = self._standardize(z_t)
        low_s, high_s = self.fd_filter(z_s)
        low_t, high_t = self.fd_filter(z_t)
        low_s = l2_normalize_feature(low_s)
        low_t = l2_normalize_feature(low_t)
        high_s = l2_normalize_feature(high_s)
        high_t = l2_normalize_feature(high_t)
        low_loss = F.mse_loss(low_s, low_t)
        high_loss = F.mse_loss(signed_log1p(high_s), signed_log1p(high_t))
        self.last_low_loss = low_loss.detach()
        self.last_high_loss = high_loss.detach()
        return self.loss_weight * (self.low_freq_weight * low_loss + self.high_freq_weight * high_loss)
