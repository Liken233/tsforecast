# -*- coding: utf-8 -*-
"""
推理核心：模型装配 + 整轨迹滚动滑窗预测。

train.py / evaluate.py / predict.py / visualize.py 共用这里，
避免各处各写一份窗口构造与反归一化，出不一致的结果。

关键约定（与 data 保持一致）：
    窗口起点 j（原始采样点下标）→
        编码器看 j, j+S, ..., j+(L-1)S 共 L 点（相对该窗口末状态的位移 Δ）
        预测未来 H 步：j+L·S ... j+(L+H-1)S，同样是相对窗口末状态的增量 Δ
        还原：绝对状态 = Δ + 窗口末状态（原始下标 j+(L-1)S）
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .data import (                                     
    DeltaNormalizer, ENC_DIM, TGT_DIM, load_matrix, make_window,
)
from .models import Seq2SeqTrajectoryNet                       


# ----------------------------------------------------------------------------
# 模型装配
# ----------------------------------------------------------------------------
def load_model_bundle(ckpt_dir: str, ckpt_name: str = "checkpoint.pth",
                      device="cpu", strict: bool = False) -> Tuple[Seq2SeqTrajectoryNet, DeltaNormalizer, dict]:
    """
    从一个训练产物目录装配 (模型, 归一化器, 配置)。

    目录内约定文件:
        config.json    训练时的完整配置
        norm.json      Δ 空间全局归一化统计
        checkpoint.pth 最优权重
    """
    ckpt_dir = Path(ckpt_dir).resolve()
    cfg_path = ckpt_dir / "config.json"
    norm_path = ckpt_dir / "norm.json"
    w_path = ckpt_dir / ckpt_name

    for p in (cfg_path, norm_path, w_path):
        if not p.exists():
            raise FileNotFoundError(f"缺少必需文件: {p}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    normalizer = DeltaNormalizer.load(str(norm_path))

    m = cfg["model"]
    model = Seq2SeqTrajectoryNet(
        input_dim=ENC_DIM,
        d_model=m["d_model"],
        num_heads=m["num_heads"],
        pred_dim=TGT_DIM,
        pred_len=cfg["data"]["pred_len"],
        layer_num=m["layer_num"],
        dropout=m["dropout"],
    )
    state = torch.load(w_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    # 兼容 DataParallel 保存的 module.* 前缀
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing or unexpected:
        print(f"[警告] 权重键不匹配: missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")
    model = model.to(device).eval()
    return model, normalizer, cfg


# ----------------------------------------------------------------------------
# 整轨迹滚动预测
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict_trajectory(
    model: Seq2SeqTrajectoryNet,
    normalizer: DeltaNormalizer,
    feat: np.ndarray,
    input_len: int,
    pred_len: int,
    data_step: int,
    device="cpu",
    horizon: Optional[int] = None,
    batch_size: int = 256,
    stride: int = 1,
    max_windows: Optional[int] = None,
    keep_ratio: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    在整条轨迹上滚动滑窗预测（每个窗口都用真值历史，开环滑窗，与训练一致）。

    返回 dict:
        full_pred   [n, H, 6]  归一化Δ还原后的绝对预测
        full_true   [n, H, 6]  绝对真值
        values      [m, 6]     指定 horizon 处的绝对预测轨迹（m ≤ n）
        truths      [m, 6]     指定 horizon 处的绝对真值
        timestamps  [m]        对应原始时间戳
        points_idx  [m]        horizon 点对应的原始采样点下标
        last_abs    [n, 6]     各窗口末状态
        start_index [n]        各窗口起点
    """
    L, H, S = int(input_len), int(pred_len), int(data_step)
    horizon = H if horizon is None else int(np.clip(horizon, 1, H))
    span = (L + H) * S
    n_total = feat.shape[0] - span
    if n_total <= 0:
        raise ValueError(f"轨迹太短（{feat.shape[0]} 点），至少需要 {span + 1} 点")

    starts = np.arange(0, n_total, max(1, int(stride)), dtype=np.int64)
    if max_windows:
        starts = starts[: int(max_windows)]
    n = starts.shape[0]

    en = np.empty((n, L, ENC_DIM), np.float32)
    de = np.empty((n, L, TGT_DIM), np.float32)
    last_abs = np.empty((n, TGT_DIM), np.float32)
    tg_delta = np.empty((n, H, TGT_DIM), np.float32)
    for i, st in enumerate(starts):
        e, d, t, la = make_window(feat, int(st), L, H, S)
        en[i] = normalizer.norm_en(e)
        dn = normalizer.norm_tg(d)
        de[i] = np.concatenate([dn, np.zeros((H, TGT_DIM), np.float32)], axis=0)
        tg_delta[i] = t
        last_abs[i] = la

    preds = np.empty((n, H, TGT_DIM), np.float64)
    for b0 in range(0, n, batch_size):
        b1 = min(b0 + batch_size, n)
        en_t = torch.from_numpy(en[b0:b1]).float().to(device)
        de_t = torch.from_numpy(de[b0:b1]).float().to(device)
        out = model(en_t, de_t).float().cpu().numpy()
        preds[b0:b1] = out

    # 归一化Δ → 物理Δ → 绝对轨迹
    full_pred = normalizer.denorm_tg(preds) + last_abs[:, None, :]
    full_true = tg_delta.astype(np.float64) + last_abs[:, None, :]

    # 指定 horizon 处的单点轨迹
    point_idx = starts + (L - 1) * S + horizon * S
    values = full_pred[:, horizon - 1, :].copy()
    truths = full_true[:, horizon - 1, :].copy()

    if keep_ratio is not None and keep_ratio < 1.0:
        keep = max(1, int(round(n * float(keep_ratio))))
        keep = min(keep, n)
        values, truths, point_idx = values[:keep], truths[:keep], point_idx[:keep]
        starts_k = starts[:keep]
    else:
        starts_k = starts

    timestamps = feat[point_idx, 0].astype(np.float64)

    return {
        "full_pred": full_pred,
        "full_true": full_true,
        "values": values,
        "truths": truths,
        "timestamps": timestamps,
        "points_idx": point_idx,
        "last_abs": last_abs,
        "start_index": starts_k,
    }


def load_trajectory(path: str, mat_key: str = "dataMatrix",
                    csv_orientation: str = "auto", csv_header: str = "auto") -> np.ndarray:
    """便捷封装：读一条轨迹 → (T, 10)。"""
    return load_matrix(path, mat_key, csv_orientation, csv_header)
