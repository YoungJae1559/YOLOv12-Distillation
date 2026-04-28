import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("FeatureAdapter", "attentionMatrix", "NonLocalAttention", "NLKD_IN_Loss")


class FeatureAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.proj = (nn.Identity() if self.in_channels == self.out_channels else nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=1, bias=bias))

    def forward(self, x):
        return self.proj(x)


class attentionMatrix(nn.Module):
    def __init__(self, in_channels, inter_channels, dimension=3, sub_sample=True):
        super().__init__()
        assert dimension in [1, 2, 3]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=2)
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1)
        if sub_sample:
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, y_s, y_t):
        assert y_s.shape == y_t.shape
        batch_size = y_s.size(0)
        theta_y_s = self.theta(y_s).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        phi_y_t = self.phi(y_t).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_y_s, phi_y_t)
        return f / f.size(-1)


class NonLocalAttention(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.att = attentionMatrix(self.in_channels, self.inter_channels, dimension, sub_sample)
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=2)
            bn = nn.BatchNorm1d
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1)
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1),
                bn(self.in_channels),
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)

    def forward(self, y_s, y_t):
        assert y_s.shape == y_t.shape
        batch_size = y_s.size(0)
        f_div_C = self.att(y_s, y_t)
        g_y_t = self.g(y_t).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        y_s_ = torch.matmul(f_div_C, g_y_t)
        y_s_ = y_s_.permute(0, 2, 1).contiguous().view(batch_size, self.inter_channels, *y_s.size()[2:])
        W_y_s = self.W(y_s_)
        z_s = W_y_s + y_s
        z_t = y_t
        return z_s, z_t


class NLKD_IN_Loss(nn.Module):
    def __init__(
        self,
        in_channels,
        inter_channels=None,
        dimension=2,
        sub_sample=True,
        bn_layer=True,
        tau=1.0,
        loss_weight=1.0,
    ):
        super().__init__()
        assert dimension in [1, 2, 3]
        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels else in_channels // 2
        if self.inter_channels == 0:
            self.inter_channels = 1
        self.tau = tau
        self.loss_weight = loss_weight
        self.non_local_att = NonLocalAttention(self.in_channels,
                                               self.inter_channels,
                                               dimension=dimension,
                                               sub_sample=sub_sample,
                                               bn_layer=bn_layer)
        if dimension == 3:
            self.in_norm = nn.InstanceNorm3d(in_channels, affine=False)
        elif dimension == 2:
            self.in_norm = nn.InstanceNorm2d(in_channels, affine=False)
        else:
            self.in_norm = nn.InstanceNorm1d(in_channels, affine=False)

    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        return self.in_norm(feat)

    def forward(self, s_feature, t_feature):
        assert s_feature.shape == t_feature.shape
        z_s, z_t = self.non_local_att(s_feature, t_feature)
        n_z_s, n_z_t = self.norm(z_s), self.norm(z_t)
        loss = F.mse_loss(n_z_s, n_z_t) / 2
        return loss * self.loss_weight
