# -*- coding: utf-8 -*-
"""
工具集：随机种子、配置读写、评估指标、早停、可视化。

图例/坐标轴文字统一使用英文，避免在无中文字体的服务器上出现方块。
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")           # 服务器无显示环境也能出图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (注册 3D 投影)

# ============================================================================
# 路径
# ============================================================================
#: 包内置默认配置
_PKG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = _PKG_DIR / "configs" / "default.json"


def resolve_path(p) -> Optional[str]:
    """把相对路径按**当前工作目录**解析为绝对路径。"""
    if p is None:
        return None
    return str(Path(str(p)).expanduser().resolve())


def resolve_data_root(p, override: Optional[str] = None, verbose: bool = True) -> str:
    """
    解析数据根目录。

    优先用 `override`（命令行显式指定），否则用配置里的路径。
    两者都会先按当前工作目录展开；若最终目录不存在，抛 FileNotFoundError
    并给出可操作的提示——checkpoint 里记录的是**训练时的机器路径**，
    换机器后需要自己用 `--data_root` 指一下。
    """
    cand = override if (override not in (None, "")) else p
    if cand in (None, ""):
        raise FileNotFoundError("未指定数据目录，请用 --data_root 指定数据根目录")

    path = Path(str(cand)).expanduser().resolve()
    if path.exists():
        if verbose and override not in (None, ""):
            print(f"[路径] 使用 --data_root 指定的数据目录: {path}")
        return str(path)

    hint = ""
    if override in (None, "") and p not in (None, ""):
        hint = (f"\n  该路径来自 checkpoint 中记录的训练配置，换机器后通常已失效。"
                f"\n  请用 --data_root 显式指定本地数据根目录。")
    raise FileNotFoundError(f"数据目录不存在: {path}{hint}")


# ============================================================================
# 基础
# ============================================================================
DIM_NAMES = ["x", "y", "z", "v", "pitch", "yaw"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"不可序列化: {type(o)}")


def deep_update(base: dict, patch: dict) -> dict:
    """递归合并配置（patch 覆盖 base）。"""
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def resolve_device(cfg: dict) -> torch.device:
    hw = cfg.get("hardware", {})
    use_gpu = bool(hw.get("use_gpu", True)) and torch.cuda.is_available()
    if use_gpu:
        gpu = int(hw.get("gpu", 0))
        if hw.get("use_multi_gpu") and str(hw.get("devices", "")).count(",") >= 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(hw.get("devices"))
            print(f"[设备] 多卡模式 CUDA_VISIBLE_DEVICES={hw.get('devices')}")
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


# ============================================================================
# 指标
# ============================================================================
def MAE(pred, true):
    return float(np.mean(np.abs(pred - true)))


def MSE(pred, true):
    return float(np.mean((pred - true) ** 2))


def RMSE(pred, true):
    return float(np.sqrt(MSE(pred, true)))


def MAPE(pred, true):
    return float(np.mean(np.abs((pred - true) / (true + 1))))


def MSPE(pred, true):
    return float(np.mean(np.square((pred - true) / (true + 1))))


def metric(pred, true):
    """与旧代码保持一致的 5 项指标，返回 (mae, mse, rmse, mape, mspe)。"""
    return MAE(pred, true), MSE(pred, true), RMSE(pred, true), MAPE(pred, true), MSPE(pred, true)


def per_dim_error(pred: np.ndarray, true: np.ndarray, names: Sequence[str] = DIM_NAMES) -> Dict[str, dict]:
    """逐维误差统计（绝对误差与相对误差）。"""
    out = {}
    for i, nm in enumerate(names[: pred.shape[-1]]):
        p, t = pred[..., i], true[..., i]
        denom = np.maximum(np.abs(t), 1e-8)
        out[nm] = {
            "mae": float(np.mean(np.abs(p - t))),
            "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
            "max_abs_err": float(np.max(np.abs(p - t))),
            "mape_pct": float(np.mean(np.abs(p - t) / denom) * 100.0),
        }
    return out


def error_by_horizon(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """按预测步长统计各维 MAE。输入 [N, H, D] → [H, D]。"""
    return np.mean(np.abs(pred - true), axis=0)


# ============================================================================
# 早停
# ============================================================================
class EarlyStopping:
    """验证损失不再下降时保存最优权重，并触发停止。"""

    def __init__(self, patience: int = 7, verbose: bool = True, delta: float = 0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss: float, model: torch.nn.Module, path: str) -> bool:
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            if self.verbose:
                print(f"  [早停] 验证损失改善: {self.val_loss_min:.6f} -> {val_loss:.6f}，保存权重")
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(val_loss, model, path)
            return True
        self.counter += 1
        if self.verbose:
            print(f"  [早停] 未改善 {self.counter}/{self.patience}")
        if self.counter >= self.patience:
            self.early_stop = True
        return False

    def save_checkpoint(self, val_loss, model, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, "checkpoint.pth"))
        self.val_loss_min = val_loss


# ============================================================================
# 可视化
# ============================================================================
def _save(fig, save_path: Optional[str], dpi: int = 200):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(history: dict, save_path: Optional[str] = None):
    """训练/验证损失 + 学习率曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    ax = axes[0]
    ax.plot(history["train_loss"], label="train", linewidth=1.8)
    ax.plot(history["val_loss"], label="val", linewidth=1.8)
    if history.get("best_epoch"):
        ax.axvline(history["best_epoch"] - 1, color="tab:red", linestyle="--",
                   linewidth=1.0, alpha=0.7, label="best")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (delta space)")
    ax.set_title("Training / validation loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(history["lr"], color="tab:green", linewidth=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_title("Learning rate schedule")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, save_path)
    return save_path


def plot_horizon_error(err_h: np.ndarray, save_path: Optional[str] = None,
                       dims: Sequence[str] = DIM_NAMES):
    """MAE vs 预测步长（默认只画位置 x/y/z）。"""
    dims = list(dims)
    idx = [dims.index(d) for d in ("x", "y", "z") if d in dims] or list(range(err_h.shape[1]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axes[0]
    for i in idx:
        ax.plot(np.arange(1, err_h.shape[0] + 1), err_h[:, i], marker="o",
                markersize=3, linewidth=1.6, label=dims[i])
    ax.set_xlabel("forecast step (x data_step)")
    ax.set_ylabel("MAE")
    ax.set_title("Position error vs horizon")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i in range(err_h.shape[1]):
        if i in idx:
            continue
        ax.plot(np.arange(1, err_h.shape[0] + 1), err_h[:, i], marker="o",
                markersize=3, linewidth=1.6, label=dims[i])
    ax.set_xlabel("forecast step (x data_step)")
    ax.set_ylabel("MAE")
    ax.set_title("v / attitude error vs horizon")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, save_path)
    return save_path


def plot_error_distribution(errors: np.ndarray, save_path: Optional[str] = None,
                            dims: Sequence[str] = DIM_NAMES):
    """各维误差分布直方图。errors: [N, H, D]"""
    d = errors.shape[-1]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.ravel()
    for i in range(min(d, 6)):
        ax = axes[i]
        ax.hist(errors[..., i].ravel(), bins=80, color="tab:blue", alpha=0.75)
        ax.axvline(0, color="tab:red", linewidth=1.0)
        ax.set_title(f"error distribution - {dims[i]}")
        ax.set_xlabel("pred - true")
        ax.set_ylabel("count")
        ax.grid(alpha=0.25)
    for j in range(min(d, 6), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    _save(fig, save_path)
    return save_path


def plot_sample_2d(history_abs: np.ndarray, true_abs: np.ndarray, pred_abs: np.ndarray,
                   save_path: Optional[str] = None, title: str = "",
                   t_hist: Optional[np.ndarray] = None, t_fut: Optional[np.ndarray] = None):
    """单样本 2D 对比：历史实线 + 未来真值虚线 + 预测点。"""
    n_h = len(history_abs)
    n_f = len(true_abs)
    if t_hist is None:
        t_hist = np.arange(n_h)
    if t_fut is None:
        t_fut = np.arange(n_h, n_h + n_f)

    dims = ["x", "y", "z", "v", "pitch", "yaw"]
    units = ["m", "m", "m", "m/s", "rad", "rad"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2))
    axes = axes.ravel()
    for i in range(6):
        ax = axes[i]
        ax.plot(t_hist, history_abs[:, i], color="tab:blue", linewidth=1.8, label="history")
        ax.axvline(t_fut[0], color="gray", linestyle=":", linewidth=1.0)
        ax.plot(t_fut, true_abs[:, i], color="tab:green", linestyle="--",
                linewidth=1.8, label="ground truth")
        ax.plot(t_fut, pred_abs[:, i], color="tab:red", linestyle="",
                marker="o", markersize=3.5, label="prediction")
        ax.set_title(f"{dims[i]}  ({units[i]})")
        ax.set_xlabel("t (s)")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _save(fig, save_path)
    return save_path


def plot_trajectory_3d(history_abs: np.ndarray, true_abs: np.ndarray, pred_abs: np.ndarray,
                       save_path: Optional[str] = None, title: str = ""):
    """3D 轨迹对比（x-y-z）。"""
    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(history_abs[:, 0], history_abs[:, 1], history_abs[:, 2],
            color="tab:blue", linewidth=2.0, label="history")
    ax.plot(true_abs[:, 0], true_abs[:, 1], true_abs[:, 2],
            color="tab:green", linestyle="--", linewidth=2.0, label="ground truth")
    ax.plot(pred_abs[:, 0], pred_abs[:, 1], pred_abs[:, 2],
            color="tab:red", linestyle=":", linewidth=2.0, label="prediction")
    ax.scatter([history_abs[-1, 0]], [history_abs[-1, 1]], [history_abs[-1, 2]],
               color="black", s=28, marker="o", label="forecast origin")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)
    return save_path


def plot_sample_grid(samples: List[dict], save_path: Optional[str] = None,
                     max_cols: int = 3, title: str = ""):
    """多样本 x 维对比网格。samples: [{history, true, pred, name}]"""
    n = len(samples)
    if n == 0:
        return None
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.6 * rows), squeeze=False)

    for k, s in enumerate(samples):
        ax = axes[k // cols][k % cols]
        h, t, p = s["history"], s["true"], s["pred"]
        th = np.arange(len(h))
        tf = np.arange(len(h), len(h) + len(t))
        ax.plot(th, h[:, 0], color="tab:blue", linewidth=1.6, label="history")
        ax.plot(tf, t[:, 0], color="tab:green", linestyle="--", linewidth=1.6, label="truth")
        ax.plot(tf, p[:, 0], color="tab:red", linestyle=":", linewidth=1.6, label="pred")
        ax.set_title(s.get("name", f"sample {k}")[:44], fontsize=9)
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=7)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _save(fig, save_path)
    return save_path


def plot_prediction_timeline(history_abs: np.ndarray, segments: List[dict],
                             save_path: Optional[str] = None, title: str = ""):
    """
    把滑窗预测拼成整条轨迹的连续时间线（用于 predict.py 的整轨迹推演图）。
    segments: [{t_fut, true, pred}]
    """
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    labels = ["x (m)", "y (m)", "z (m)"]
    for i, ax in enumerate(axes):
        ax.plot(np.arange(len(history_abs)), history_abs[:, i],
                color="tab:blue", linewidth=1.8, label="history")
        for si, seg in enumerate(segments):
            ax.plot(seg["t_fut"], seg["true"][:, i], color="tab:green",
                    linestyle="--", linewidth=1.2)
            ax.plot(seg["t_fut"], seg["pred"][:, i], color="tab:red",
                    linestyle=":", linewidth=1.2, marker="o", markersize=2.5,
                    label=("prediction" if si == 0 else None))
        ax.axvline(len(history_abs), color="gray", linestyle=":", linewidth=1.0)
        ax.set_ylabel(labels[i])
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)
    axes[-1].set_xlabel("sampled step index")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _save(fig, save_path)
    return save_path
