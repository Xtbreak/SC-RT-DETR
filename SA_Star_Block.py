# Extracted from the customized Ultralytics fork (AGPL-3.0).
"""Spatial-Aware Improved Star Block, used in the Star-Chain Network.

Requires PyTorch and timm. Preserves the local ReLU6 star interaction.
"""

import torch
import torch.nn as nn
from timm.models.layers import DropPath

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)

class SpatialAttentionLite(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.max(x, dim=1, keepdim=True)[0]
        P = torch.cat((avg_map, max_map), dim=1)
        # Eq(4): M = sigma(phi(eta(Conv_7x7(P))))，eta=BN, phi=ReLU
        M = self.sigmoid(self.relu(self.bn(self.conv(P))))
        return M

class SAStarBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.spatial_attn = SpatialAttentionLite(kernel_size=7)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x_star = self.act(self.f1(x)) * self.f2(x)
        x_star = x_star * self.spatial_attn(x_star)
        x = self.dwconv2(self.g(x_star))
        return identity + self.drop_path(x)
