#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速上手：用 Python API 走一遍「造数据 → 建数据集 → 训练 → 评估 → 出图」。

比命令行更灵活，便于嵌到自己的实验脚本里。

用法:
    python examples/make_synthetic_data.py --n-files 6 --n-points 12000
    python examples/quickstart.py
    python examples/quickstart.py --data-root ./data/dataset --epochs 8 --d-model 128
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tsforecast.data import build_loaders, ENC_DIM, TGT_DIM, load_matrix
from tsforecast.inference import load_model_bundle, predict_trajectory
from tsforecast.models import Seq2SeqTrajectoryNet
from tsforecast.utils import (
    dump_json, metric, plot_horizon_error, plot_sample_2d, resolve_device,
    set_seed, error_by_horizon,
)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="tsforecast 快速上手")
    ap.add_argument("--data-root", type=str, default="./data/dataset")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layer-num", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", type=str, default=None,
                    help="产物目录（默认写到临时目录，不污染仓库）")
    args = ap.parse_args(argv)

    out_root = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="tsforecast_quick_"))
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[输出] {out_root}")

    # ---------------------------------------------------------------- 配置
    cfg = {
        "data": {
            "data_root": args.data_root,
            "datasets": [],
            "file_types": [".mat", ".csv", ".npy"],
            "mat_key": "dataMatrix",
            "csv_orientation": "auto",
            "csv_header": "auto",
            "input_len": 40,
            "pred_len": 20,
            "data_step": 50,
            "samples_per_file": 120,
            "cache_size": 4,
            "batch_size": args.batch_size,
            "num_workers": 0,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "split_by": "file",
            "seed": 42,
        },
        "model": {"d_model": args.d_model, "num_heads": 4,
                  "layer_num": args.layer_num, "dropout": 0.1},
        "train": {"epochs": args.epochs, "learning_rate": 3e-4, "weight_decay": 0.01,
                  "grad_clip": 1.0, "use_amp": True},
        "hardware": {"use_gpu": True, "gpu": 0, "use_multi_gpu": False},
    }

    set_seed(cfg["data"]["seed"])
    device = resolve_device(cfg)
    print(f"[设备] {device}")

    # ---------------------------------------------------------------- 数据
    built = build_loaders(cfg, verbose=True)
    loaders, normalizer = built["loaders"], built["normalizer"]
    print(f"[数据] train={len(built['datasets']['train'])}  "
          f"val={len(built['datasets']['val'])}  test={len(built['datasets']['test'])} 窗口")

    # ---------------------------------------------------------------- 模型
    model = Seq2SeqTrajectoryNet(
        input_dim=ENC_DIM,
        d_model=cfg["model"]["d_model"],
        num_heads=cfg["model"]["num_heads"],
        pred_dim=TGT_DIM,
        pred_len=cfg["data"]["pred_len"],
        layer_num=cfg["model"]["layer_num"],
        dropout=cfg["model"]["dropout"],
    ).to(device).float()
    print(f"[模型] 参数量 {model.n_params() / 1e6:.3f} M")
    print(f"[预测目标] Δ 增量，{TGT_DIM} 维；推理时按 Δ + 窗口末状态还原绝对值")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"],
                                  weight_decay=cfg["train"]["weight_decay"])
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ---------------------------------------------------------------- 训练
    best_val, best_state = float("inf"), None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for en, de, tg, _last in loaders["train"]:
            en, de, tg = en.to(device), de.to(device), tg.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = criterion(model(en, de), tg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item()

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for en, de, tg, _last in loaders["val"]:
                en, de, tg = en.to(device), de.to(device), tg.to(device)
                va_loss += criterion(model(en, de), tg).item()
        tr_loss /= max(len(loaders["train"]), 1)
        va_loss /= max(len(loaders["val"]), 1)
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:>3}/{args.epochs} | train {tr_loss:.6f} | val {va_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[训练] 用时 {time.time() - t0:.1f}s，最优验证损失 {best_val:.6f}")

    # ---------------------------------------------------------------- 评估
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for en, de, tg, last in loaders["test"]:
            p = model(en.to(device), de.to(device)).cpu().numpy()
            preds.append(normalizer.to_absolute(p, last.numpy()))
            trues.append(normalizer.to_absolute(tg.numpy(), last.numpy()))
    pred = np.concatenate(preds, 0)
    true = np.concatenate(trues, 0)
    mae, mse, rmse, mape, mspe = metric(pred, true)
    print(f"[测试] shape={pred.shape}  MAE={mae:.4f}  RMSE={rmse:.4f}")

    # ---------------------------------------------------------------- 产物
    ckpt_dir = out_root / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "checkpoint.pth")
    normalizer.save(ckpt_dir / "norm.json")
    cfg_saved = dict(cfg, data=dict(cfg["data"], data_root=str(Path(args.data_root).resolve())))
    dump_json(cfg_saved, ckpt_dir / "config.json")
    dump_json({"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "mspe": mspe,
               "shape": list(pred.shape)}, out_root / "metrics.json")

    plot_horizon_error(error_by_horizon(pred, true), out_root / "error_vs_horizon.png")
    s0 = "sample0"
    plot_sample_2d(np.zeros((40, TGT_DIM)), true[0], pred[0],
                   out_root / "sample_2d.png", title=s0)
    print(f"[产物] {out_root}")

    # ---------------------------------------------------------------- 推理演示
    files = sorted(Path(args.data_root).rglob("*.mat")) or sorted(Path(args.data_root).rglob("*.csv"))
    if files:
        feat = load_matrix(str(files[0]))
        m, norm2, _ = load_model_bundle(str(ckpt_dir), device=device)
        res = predict_trajectory(m, norm2, feat,
                                 input_len=cfg["data"]["input_len"],
                                 pred_len=cfg["data"]["pred_len"],
                                 data_step=cfg["data"]["data_step"],
                                 device=device, max_windows=200)
        print(f"[推理] {files[0].name}: 预测 {len(res['points_idx'])} 个点，"
              f"首窗口末状态 {np.round(res['last_abs'][0], 2)}")

    print("\n完成。用命令行跑完整流程: tsforecast-train --help")


if __name__ == "__main__":
    main()
