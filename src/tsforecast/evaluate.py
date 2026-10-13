#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估入口：飞行器时序状态预测（增量 Δ 空间）。

加载训练产物（checkpoint.pth + config.json + norm.json + split.json），
在**同一批留出文件**上复现评估，输出指标与可视化。

用法:
    tsforecast-eval --checkpoint_dir ./checkpoint/<setting>
    tsforecast-eval --checkpoint_dir <dir> --split val
    tsforecast-eval --checkpoint_dir <dir> --data_root ./data/dataset   # 换机器后用
    tsforecast-eval --checkpoint_dir <dir> --max_files 2 --batch_size 64  # 快速看

产物（默认写到 <result_root>/<setting>/eval/，或 --out_dir 指定）:
    metrics.json            总体 + 逐维 + 分步长指标
    error_vs_horizon.png    MAE 随预测步长的变化
    error_distribution.png  各维误差分布
    sample_grid.png         多样本对比
    sample_trajectory_2d.png / sample_trajectory_3d.png
    pred.npy / true.npy     绝对状态预测与真值
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


from tsforecast.data import build_loaders, DeltaNormalizer, ENC_DIM, TGT_DIM
from tsforecast.models import Seq2SeqTrajectoryNet                                
from tsforecast.utils import (                                                    
    dump_json, error_by_horizon, load_json, metric, per_dim_error,
    plot_error_distribution, plot_horizon_error, plot_sample_grid, plot_sample_2d,
    plot_trajectory_3d, resolve_data_root, resolve_device, resolve_path, set_seed,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="飞行器时序状态预测 - 评估")
    p.add_argument("--checkpoint_dir", type=str, required=True,
                   help="训练产出的目录（含 checkpoint.pth / config.json / norm.json / split.json）")
    p.add_argument("--ckpt_name", type=str, default="checkpoint.pth")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--data_root", type=str, default=None,
                   help="覆盖 checkpoint 里记录的数据根目录（换机器时用）")
    p.add_argument("--datasets", type=str, default=None,
                   help="只使用这些子目录，逗号分隔；留空=沿用训练时的选择")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--n_samples", type=int, default=6, help="可视化展示的样本数")
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gpu", type=int, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ckpt_dir = Path(args.checkpoint_dir).resolve()
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"目录不存在: {ckpt_dir}")

    cfg = load_json(ckpt_dir / "config.json")
    normalizer = DeltaNormalizer.load(str(ckpt_dir / "norm.json"))
    split_manifest = load_json(ckpt_dir / "split.json")

    # checkpoint 里记的是训练时的路径，换机器后用 --data_root 覆盖
    cfg["data"]["data_root"] = resolve_data_root(cfg["data"]["data_root"],
                                                 override=args.data_root)
    if args.datasets:
        cfg["data"]["datasets"] = [s.strip() for s in args.datasets.replace(";", ",").split(",")
                                   if s.strip()]

    if args.cpu:
        cfg["hardware"]["use_gpu"] = False
    if args.gpu is not None:
        cfg["hardware"]["gpu"] = args.gpu
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    cfg["data"]["num_workers"] = args.num_workers

    setting = ckpt_dir.name
    out_dir = (Path(resolve_path(args.out_dir)).resolve() if args.out_dir else (
        Path(resolve_path(cfg["train"]["result_root"])) / setting / "eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg)
    set_seed(int(cfg["data"]["seed"]))

    print("=" * 78)
    print(f"[评估] {setting}")
    print(f"[权重] {ckpt_dir / args.ckpt_name}")
    print(f"[数据] root={cfg['data']['data_root']}  split={args.split}")
    print(f"[输出] {out_dir}")
    print("=" * 78)

    built = build_loaders(cfg, split_manifest=split_manifest,
                          max_files=args.max_files, verbose=True)
    loaders = built["loaders"]
    loader = loaders[args.split]
    ds = built["datasets"][args.split]

    m = cfg["model"]
    model = Seq2SeqTrajectoryNet(
        input_dim=ENC_DIM, d_model=m["d_model"], num_heads=m["num_heads"],
        pred_dim=TGT_DIM, pred_len=cfg["data"]["pred_len"],
        layer_num=m["layer_num"], dropout=m["dropout"],
    )
    state = torch.load(ckpt_dir / args.ckpt_name, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[警告] 权重键不匹配: missing={missing[:4]} unexpected={unexpected[:4]}")
    model = model.to(device).eval()
    print(f"[模型] 参数量 {model.n_params() / 1e6:.3f} M  (已加载最优权重)")

    # ---- 推理 ----
    criterion = nn.MSELoss()
    preds, trues, norm_loss, nb = [], [], 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = [t.to(device, non_blocking=True) for t in batch]
            pred = model(batch[0], batch[1])
            norm_loss += criterion(pred, batch[2]).item()
            nb += 1
            last = batch[3].float().cpu().numpy()[:, None, :]
            preds.append(normalizer.denorm_tg(pred.float().cpu().numpy()) + last)
            trues.append(normalizer.denorm_tg(batch[2].float().cpu().numpy()) + last)

    pred_abs = np.concatenate(preds, axis=0)
    true_abs = np.concatenate(trues, axis=0)

    mae, mse, rmse, mape, mspe = metric(pred_abs, true_abs)
    err_h = error_by_horizon(pred_abs, true_abs)
    err = pred_abs - true_abs
    pos = pred_abs[..., :3] - true_abs[..., :3]
    dist_err = np.linalg.norm(pos, axis=-1)          # 每个预测点的空间距离误差

    summary = {
        "setting": setting,
        "split": args.split,
        "n_windows": int(pred_abs.shape[0]),
        "horizon": int(pred_abs.shape[1]),
        "normalized_mse": norm_loss / max(nb, 1),
        "mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "mspe": mspe,
        "position_mae_m": float(np.mean(np.abs(pos))),
        "position_rmse_m": float(np.sqrt(np.mean(pos ** 2))),
        "position_distance_err_mean_m": float(dist_err.mean()),
        "position_distance_err_p90_m": float(np.percentile(dist_err, 90)),
        "last_step_position_mae_m": float(np.mean(np.abs(pos[:, -1, :]))),
        "per_dim": per_dim_error(pred_abs, true_abs),
        "mae_by_horizon": err_h.tolist(),
    }

    print("=" * 78)
    print(f"[结果] {args.split} 集 {pred_abs.shape[0]:,} 个窗口 × {pred_abs.shape[1]} 步")
    print(f"  MAE {mae:.6f}   MSE {mse:.6f}   RMSE {rmse:.6f}")
    print(f"  位置 MAE              : {summary['position_mae_m']:.3f} m")
    print(f"  空间距离误差 均值     : {summary['position_distance_err_mean_m']:.3f} m")
    print(f"  空间距离误差 P90      : {summary['position_distance_err_p90_m']:.3f} m")
    print(f"  末步位置 MAE          : {summary['last_step_position_mae_m']:.3f} m")
    print("  ---- 逐维 ----")
    for k, v in summary["per_dim"].items():
        print(f"    {k:<6s} MAE {v['mae']:>12.4f}   RMSE {v['rmse']:>12.4f}   "
              f"最大误差 {v['max_abs_err']:>12.4f}")
    print("=" * 78)

    dump_json(summary, out_dir / "metrics.json")
    np.save(out_dir / "pred.npy", pred_abs)
    np.save(out_dir / "true.npy", true_abs)

    # ---- 可视化 ----
    plot_horizon_error(err_h, out_dir / "error_vs_horizon.png")
    plot_error_distribution(err, out_dir / "error_distribution.png")

    n_show = max(1, min(args.n_samples, len(ds)))
    samples = []
    for i in range(n_show):
        raw = ds.raw_absolute(i)
        samples.append({
            "history": raw["history_abs"],
            "true": raw["future_abs"],
            "pred": pred_abs[i],
            "name": Path(raw["file"]).stem,
        })
    plot_sample_grid(samples, out_dir / "sample_grid.png",
                     title=f"{setting} - prediction (red) vs truth (green), x axis")
    s0 = samples[0]
    plot_sample_2d(s0["history"], s0["true"], s0["pred"],
                   out_dir / "sample_trajectory_2d.png", title=s0["name"])
    plot_trajectory_3d(s0["history"], s0["true"], s0["pred"],
                       out_dir / "sample_trajectory_3d.png", title=s0["name"])

    print(f"[完成] 指标与图已保存到: {out_dir}")


if __name__ == "__main__":
    main()
