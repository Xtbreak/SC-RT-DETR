# Extracted from the customized Ultralytics fork (AGPL-3.0).
"""Multi-Scale Heterogeneous Kernel Selection Mechanism (MHKS).

HCFM uses kernel sets (1, 3, 5), (3, 5, 7), (5, 7, 9) at P3, P4, P5.
Kernel sets are predefined per level, not selected dynamically.
Requires the customized Ultralytics framework and PyTorch.
"""

import math
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f

class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, dw_parallel=True):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                Conv(self.in_channels, self.in_channels, kernel_size, s=stride, g=self.in_channels)
            )
            for kernel_size in self.kernel_sizes
        ])

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x+dw_out
        # You can return outputs based on what you intend to do with them
        return outputs

class MHKS(nn.Module):
    """
    Multi-scale convolution block (MHKS)
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[1,3,5], stride=1, expansion_factor=2, dw_parallel=True, add=True):
        super(MHKS, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.n_scales = len(self.kernel_sizes)
        # check stride value
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            Conv(self.in_channels, self.ex_channels, 1)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels*1
        else:
            self.combined_channels = self.ex_channels*self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            Conv(self.combined_channels, self.out_channels, 1, act=False)
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add == True:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = self.channel_shuffle(dout, math.gcd(self.combined_channels,self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

class CSP_MHKS(C2f):
    def __init__(self, c1, c2, n=1, kernel_sizes=[1,3,5], shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)

        self.m = nn.ModuleList(MHKS(self.c, self.c, kernel_sizes=kernel_sizes) for _ in range(n))
