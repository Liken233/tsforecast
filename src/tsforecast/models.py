# -*- coding: utf-8 -*-
"""
模型定义（自包含，无第三方依赖，仅 torch）。

结构: 编码器-解码器 Transformer（Informer 式直接多步预测头）

    编码器输入 en   : [B, L, 9]   相对窗口末状态的位移 Δ（已归一化）
    解码器输入 de   : [B, L, 6]   = [已知前缀(末 H 步的 Δ)，预测位补 0]
    输出 pred       : [B, H, 6]   未来 H 步的 Δ（已归一化）

模块命名遵循 Informer 原版约定（encode_embed / decode_embed / encoder / decoder / fc
及其内部层），**键名保持稳定**，因此历史版本的 state_dict 可以直接加载。

注意：本文件不做任何绝对量还原，输出始终是归一化增量 Δ，
      绝对序列由 data.DeltaNormalizer.to_absolute() 负责还原。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 因果掩码
# ============================================================================
class TriangularCausalMask:
    """上三角为 True 的因果掩码（Informer 同款）。"""

    def __init__(self, B: int, L: int, device="cpu"):
        shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(shape, dtype=torch.bool, device=device), diagonal=1
            )

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


# ============================================================================
# 嵌入层
# ============================================================================
class PositionEmbedding(nn.Module):
    """正弦位置编码。"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[: x.size(1), :].unsqueeze(0)


class TokenEmbedding(nn.Module):
    """卷积式值嵌入。"""

    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, out_channels=d_model,
            kernel_size=3, padding=1, padding_mode="circular",
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tokenConv(x.permute(0, 2, 1)).transpose(2, 1)


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))


# ============================================================================
# 注意力
# ============================================================================
class FullAttention(nn.Module):
    def __init__(self, attention_dropout: float = 0.1, if_mask: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.if_mask = if_mask

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.if_mask:
            attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores = scores.masked_fill(attn_mask.mask, -1e4)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attention_dropout: float = 0.1,
                 if_mask: bool = False):
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.inner_attention = FullAttention(attention_dropout=attention_dropout, if_mask=if_mask)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        q = self.query_projection(queries).view(B, L, H, -1)
        k = self.key_projection(keys).view(B, S, H, -1)
        v = self.value_projection(values).view(B, S, H, -1)
        out = self.inner_attention(q, k, v).view(B, L, -1)
        return self.out_projection(out)


# ============================================================================
# 编码器 / 解码器
# ============================================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int = 120, n_head: int = 6, dropout: float = 0.1):
        super().__init__()
        d_ff = 4 * d_model
        self.attention = AttentionLayer(d_model=d_model, n_heads=n_head,
                                        attention_dropout=dropout, if_mask=False)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, x):
        x = x + self.dropout(self.attention(x, x, x))

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, d_model: int, n_head: int = 6, layer_num: int = 1, dropout: float = 0.1):
        super().__init__()
        self.attn_layers = nn.ModuleList(
            [EncoderLayer(d_model=d_model, n_head=n_head, dropout=dropout) for _ in range(layer_num)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for attn_layer in self.attn_layers:
            x = attn_layer(x)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int = 120, n_head: int = 6, dropout: float = 0.1):
        super().__init__()
        d_ff = 4 * d_model
        self.attention = AttentionLayer(d_model=d_model, n_heads=n_head,
                                        attention_dropout=dropout, if_mask=True)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, x, cross):
        x = x + self.dropout(self.attention(x, x, x))
        y = x = self.norm1(x)

        x = x + self.dropout(self.attention(x, cross, cross))
        y = x = self.norm2(x)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, d_model: int, n_head: int = 6, layer_num: int = 1, dropout: float = 0.1):
        super().__init__()
        self.attn_layers = nn.ModuleList(
            [DecoderLayer(d_model=d_model, n_head=n_head, dropout=dropout) for _ in range(layer_num)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, cross):
        for attn_layer in self.attn_layers:
            x = attn_layer(x, cross)
        return self.norm(x)


# ============================================================================
# 主模型
# ============================================================================
class Seq2SeqTrajectoryNet(nn.Module):
    """
    轨迹增量预测网络（编码器-解码器 Transformer）。

    参数:
        input_dim : 编码器输入维度（本包固定 9）
        d_model   : 隐层维度
        num_heads : 注意力头数
        pred_dim  : 预测维度（本包固定 6: x,y,z,v,pitch,yaw）
        pred_len  : 预测步数 H
        layer_num : 编码器/解码器层数
        dropout   : dropout 比例
    """

    def __init__(self, input_dim: int = 9, d_model: int = 256, num_heads: int = 8,
                 pred_dim: int = 6, pred_len: int = 20, layer_num: int = 6,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.pred_dim = pred_dim
        self.pred_len = pred_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.layer_num = layer_num
        self.dropout_rate = dropout

        # 特征嵌入
        self.encode_embed = DataEmbedding(input_dim, d_model)
        self.decode_embed = DataEmbedding(pred_dim, d_model)

        # 编码器 / 解码器
        self.encoder = Encoder(d_model=d_model, n_head=num_heads,
                               layer_num=layer_num, dropout=dropout)
        self.decoder = Decoder(d_model=d_model, n_head=num_heads,
                               layer_num=layer_num, dropout=dropout)

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(d_model, 2 * pred_dim),
            nn.ReLU(),
            nn.Linear(2 * pred_dim, pred_dim),
        )

    def forward(self, en_input: torch.Tensor, de_input: torch.Tensor) -> torch.Tensor:
        """en_input [B, L, input_dim]，de_input [B, L, pred_dim] → [B, pred_len, pred_dim]"""
        encoded = self.encoder(self.encode_embed(en_input))
        decoded = self.decoder(self.decode_embed(de_input), encoded)
        return self.fc(decoded[:, -self.pred_len:, :])

    # 便捷方法 -------------------------------------------------------------
    def arch_kwargs(self) -> dict:
        return dict(input_dim=self.input_dim, d_model=self.d_model, num_heads=self.num_heads,
                    pred_dim=self.pred_dim, pred_len=self.pred_len, layer_num=self.layer_num,
                    dropout=self.dropout_rate)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 兼容旧代码/旧权重里的类名
model2 = Seq2SeqTrajectoryNet

__all__ = [
    "Seq2SeqTrajectoryNet", "model2", "DataEmbedding", "AttentionLayer",
    "Encoder", "Decoder", "EncoderLayer", "DecoderLayer", "TriangularCausalMask",
]
