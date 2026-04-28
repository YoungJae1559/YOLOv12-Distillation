import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("PlainFeatureAdapter", "PlainFeatureKDLoss")


class PlainFeatureAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.proj = nn.Identity() if self.in_channels == self.out_channels else nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class PlainFeatureKDLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = float(loss_weight)

    def forward(self, s_feature, t_feature):
        if s_feature.shape != t_feature.shape:
            raise RuntimeError(f"Plain KD feature shape mismatch: {tuple(s_feature.shape)} vs {tuple(t_feature.shape)}")
        return self.loss_weight * F.mse_loss(s_feature, t_feature)
