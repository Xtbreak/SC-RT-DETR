# Extracted from the customized Ultralytics fork (AGPL-3.0).
"""Smoothing Learned 2D Positional Encoding (SLPE).

Preserves the local implementation: zero base vectors and step=0.01.
Requires the customized Ultralytics framework and PyTorch.
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.transformer import TransformerEncoderLayer

class SLPE(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length, delta=0.01):
        super(SLPE, self).__init__()
        # 计算2D网格的宽高 (假设是方形)
        self.h = self.w = int(seq_length ** 0.5)
        assert self.h * self.w == seq_length, f"seq_length {seq_length} must be a perfect square"

        d = embedding_dim // 2

        # 分别为x和y坐标创建embedding
        self.pe_x = nn.Embedding(self.w, d)
        self.pe_y = nn.Embedding(self.h, d)

        # SLPE: 线性平滑初始化，对应论文公式(7)
        # E_i^x = b_x + i * delta * 1_d  (b_x 取 0)
        # E_j^y = b_y + j * delta * 1_d  (b_y 取 0)
        with torch.no_grad():
            self.pe_x.weight.data.copy_(
                torch.arange(self.w, dtype=torch.float).unsqueeze(1).expand(-1, d) * delta
            )
            self.pe_y.weight.data.copy_(
                torch.arange(self.h, dtype=torch.float).unsqueeze(1).expand(-1, d) * delta
            )

        # 预计算2D坐标网格
        x_coords = torch.arange(self.w).repeat(self.h, 1)  # [H, W]
        y_coords = torch.arange(self.h).repeat_interleave(self.w).view(self.h, self.w)  # [H, W]

        # 展平并注册为buffer
        self.register_buffer("x_position_ids", x_coords.flatten().unsqueeze(0))  # [1, H*W]
        self.register_buffer("y_position_ids", y_coords.flatten().unsqueeze(0))  # [1, H*W]

    def forward(self, x, position_ids=None):
        seq_len = x.size(-1) if len(x.shape) == 4 else x.size(1)  # 兼容不同输入格式

        # 使用预计算的坐标或传入的坐标
        if position_ids is None:
            x_ids = self.x_position_ids[:, :seq_len]
            y_ids = self.y_position_ids[:, :seq_len]
        else:
            # 如果传入自定义position_ids，需要拆分成x,y坐标
            x_ids = position_ids % self.w
            y_ids = position_ids // self.w

        # 分别获取x和y的位置编码
        x_embeddings = self.pe_x(x_ids)  # [1, seq_len, embedding_dim//2]
        y_embeddings = self.pe_y(y_ids)  # [1, seq_len, embedding_dim//2]

        # 拼接x和y的编码
        position_embeddings = torch.cat([x_embeddings, y_embeddings], dim=-1)  # [1, seq_len, embedding_dim]

        return position_embeddings

class AIFI_SLPE(TransformerEncoderLayer):
    """AIFI transformer layer with Smoothing Learned 2D Positional Encoding (SLPE)."""

    def __init__(self, c1, cm=2048, num_heads=8, fmap_size=20 * 20, dropout=0,
                 act=nn.GELU(), normalize_before=False, step=0.01):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        self.LPE = SLPE(fmap_size, c1, fmap_size, delta=step)

    def forward(self, x):
        c, h, w = x.shape[1:]
        x_flat = x.flatten(2).permute(0, 2, 1)          # [B, H*W, C]
        pos_embed = self.LPE(x_flat).to(device=x.device, dtype=x.dtype)
        x_out = super().forward(x_flat + pos_embed)
        return x_out.permute(0, 2, 1).view([-1, c, h, w]).contiguous()


globals()['AIFI_SLPE'] = AIFI_SLPE
