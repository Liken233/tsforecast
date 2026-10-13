# -*- coding: utf-8 -*-
"""模型测试：前向形状 / 因果性 / 梯度 / 参数量 / state_dict 键名。"""

from __future__ import annotations

import pytest
import torch

from tsforecast.data import ENC_DIM, TGT_DIM
from tsforecast.models import (
    Seq2SeqTrajectoryNet, model2, FullAttention, TriangularCausalMask, DataEmbedding,
)


def _model(L=40, H=20, d=32, layers=2, heads=4):
    return Seq2SeqTrajectoryNet(input_dim=ENC_DIM, d_model=d, num_heads=heads,
                                pred_dim=TGT_DIM, pred_len=H, layer_num=layers,
                                dropout=0.0)


def test_别名兼容():
    assert model2 is Seq2SeqTrajectoryNet


def test_forward_输出形状():
    m = _model()
    B, L, H = 5, 40, 20
    en = torch.randn(B, L, ENC_DIM)
    de = torch.randn(B, L, TGT_DIM)
    out = m(en, de)
    assert out.shape == (B, H, TGT_DIM)


@pytest.mark.parametrize("L,H", [(40, 20), (60, 10), (32, 32)])
def test_forward_窗口长度可变(L, H):
    m = _model(L=L, H=H)
    out = m(torch.randn(3, L, ENC_DIM), torch.randn(3, L, TGT_DIM))
    assert out.shape == (3, H, TGT_DIM)


def test_pred_len_与权重无关():
    """逐步输出头 + 固定位置编码 ⇒ 换 pred_len 仍能加载同一份 state_dict。"""
    m20 = _model(H=20, d=64)
    sd = m20.state_dict()
    m30 = _model(H=30, d=64)
    m30.load_state_dict(sd, strict=True)


def test_参数量随层数与宽度增长():
    assert _model(d=32, layers=1).n_params() < _model(d=64, layers=1).n_params()
    assert _model(d=32, layers=1).n_params() < _model(d=32, layers=3).n_params()
    assert _model().n_params() > 0


def test_arch_kwargs_自洽():
    kw = _model().arch_kwargs()
    rebuilt = Seq2SeqTrajectoryNet(**kw)
    assert rebuilt.pred_len == kw["pred_len"] and rebuilt.d_model == kw["d_model"]


def test_state_dict_键名与旧实现一致():
    """键名必须稳定，否则旧权重无法热启动。"""
    keys = set(_model().state_dict().keys())
    for must in (
        "encode_embed.value_embedding.tokenConv.weight",
        "encode_embed.position_embedding.pe",
        "decode_embed.value_embedding.tokenConv.weight",
        "encoder.attn_layers.0.attention.query_projection.weight",
        "decoder.attn_layers.0.attention.out_projection.weight",
        "fc.0.weight",
        "fc.2.weight",
    ):
        assert must in keys, f"缺少键 {must}"


def test_fc_逐点输出形状():
    m = _model(d=64, H=20)
    # Linear(d_model, 2*pred_dim) -> ReLU -> Linear(2*pred_dim, pred_dim)
    assert m.fc[0].weight.shape == (2 * TGT_DIM, 64)
    assert m.fc[2].weight.shape == (TGT_DIM, 2 * TGT_DIM)


def test_梯度可回传():
    m = _model()
    en = torch.randn(4, 40, ENC_DIM, requires_grad=True)
    de = torch.randn(4, 40, TGT_DIM)
    m(en, de).sum().backward()
    assert en.grad is not None and torch.isfinite(en.grad).all()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_因果掩码_上三角为真():
    """Informer 约定：上三角（未来位置）为 True，表示被屏蔽。"""
    mask = TriangularCausalMask(4, 4).mask
    assert mask.shape == (4, 1, 4, 4)
    assert mask.dtype == torch.bool
    # 对角线上及以下可看
    assert bool(mask[0, 0, 0, 0]) is False
    assert bool(mask[0, 0, 3, 2]) is False
    # 严格上三角（未来）屏蔽
    assert bool(mask[0, 0, 0, 1]) is True
    assert bool(mask[0, 0, 0, 3]) is True


def test_注意力输出形状():
    att = FullAttention(attention_dropout=0.0, if_mask=False)
    q = torch.randn(2, 5, 4, 8)          # [B, L, heads, d_k]
    out = att(q, q, q)
    assert out.shape == q.shape

    masked = FullAttention(attention_dropout=0.0, if_mask=True)
    assert masked(q, q, q).shape == q.shape


def test_嵌入形状():
    emb = DataEmbedding(c_in=ENC_DIM, d_model=64)
    assert emb(torch.randn(3, 40, ENC_DIM)).shape == (3, 40, 64)


def test_eval_模式确定性():
    m = _model().eval()
    en, de = torch.randn(2, 40, ENC_DIM), torch.randn(2, 40, TGT_DIM)
    with torch.no_grad():
        assert torch.allclose(m(en, de), m(en, de))
