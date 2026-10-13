#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理入口：飞行器时序状态预测（增量 Δ 空间）。

对整条状态序列做滚动滑窗预测，出图并导出结果。

用法:
    # 指定单个数据文件
    tsforecast-predict --checkpoint_dir ./checkpoint/<setting> \
        --data_file ./data/dataset/flightA/001.mat

    # 用目录 + 序号挑选
    tsforecast-predict --checkpoint_dir <dir> --data_root ./data/dataset \
        --datasets "flightA" --file_index 3

    # 只跑前 200 个窗口、每一步都开窗
    tsforecast-predict --checkpoint_dir <dir> --data_file <path> --stride 1 --windows 200

说明:
    每个窗口都用「真值历史」作为输入（开环滑窗），与训练的样本构造完全一致。

产物（默认 results/<setting>/predict）:
    prediction.npz      full_pred / full_true / values / truths / timestamps
    timeline.png        整段 x/y/z 时间线（真值 vs 预测）
    trajectory_3d.png   整段 3D 对比
    first_window_2d.png 首个窗口 6 维对比
    error_vs_horizon.png MAE 随预测步长
    metrics.json        误差统计
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


from tsforecast.data import discover_datasets, load_matrix, TGT_DIM
from tsforecast.inference import load_model_bundle, predict_trajectory
from tsforecast.utils import (
    dump_json, error_by_horizon, metric, per_dim_error, plot_horizon_error,
    plot_prediction_timeline, plot_sample_2d, plot_trajectory_3d,
    resolve_data_root, resolve_path,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="飞行器时序状态预测 - 推理")
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--ckpt_name", type=str, default="checkpoint.pth")
    p.add_argument("--data_file", type=str, default=None, help="直接指定数据文件（.mat/.csv/.npy）")
    p.add_argument("--data_root", type=str, default=None, help="或给数据根目录")
    p.add_argument("--datasets", type=str, default=None, help="子目录，逗号分隔；留空=全部")
    p.add_argument("--file_index", type=int, default=0, help="在候选文件中的序号")
    p.add_argument("--horizon", type=int, default=None, help="展示第几步预测（默认=pred_len）")
    p.add_argument("--stride", type=int, default=1, help="窗口滑动步长（原始采样点）")
    p.add_argument("--windows", type=int, default=None, help="最多用几个窗口（默认全部）")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--input_len", type=int, default=None)
    p.add_argument("--pred_len", type=int, default=None)
    p.add_argument("--data_step", type=int, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gpu", type=int, default=None)
    return p.parse_args(argv)


def resolve_file(args, cfg) -> Path:
    if args.data_file:
        fp = Path(args.data_file)
        if not fp.is_file():
            raise FileNotFoundError(f"数据文件不存在: {fp}")
        return fp.resolve()

    data_root = resolve_data_root(cfg["data"]["data_root"], override=args.data_root)
    ds_arg = args.datasets
    datasets = [s for s in ds_arg.replace(";", ",").split(",") if s.strip()] if ds_arg else None
    mapping = discover_datasets(data_root, datasets, tuple(cfg["data"]["file_types"]))
    files = [p for v in mapping.values() for p in v]
    if not files:
        raise FileNotFoundError(f"{data_root} 下没有数据文件")
    idx = max(0, min(args.file_index, len(files) - 1))
    print(f"[数据] 候选 {len(files)} 个文件，选用 #{idx}: {files[idx]}")
    return files[idx]


def main(argv=None):
    args = parse_args(argv)
    ckpt_dir = Path(args.checkpoint_dir).resolve()

    if args.cpu:
        device = torch.device("cpu")
    elif args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, normalizer, cfg = load_model_bundle(ckpt_dir, args.ckpt_name, device=device)

    L = int(args.input_len or cfg["data"]["input_len"])
    H = int(args.pred_len or cfg["data"]["pred_len"])
    S = int(args.data_step or cfg["data"]["data_step"])

    out_dir = (Path(resolve_path(args.out_dir)).resolve() if args.out_dir else (
        Path(resolve_path(cfg["train"]["result_root"])) / ckpt_dir.name / "predict"))
    out_dir.mkdir(parents=True, exist_ok=True)

    data_file = resolve_file(args, cfg)
    feat = load_matrix(data_file, cfg["data"]["mat_key"],
                       cfg["data"]["csv_orientation"], cfg["data"]["csv_header"])
    print(f"[数据] {data_file.name}  形状 {feat.shape} (时间 × 特征)")
    print(f"[模型] {ckpt_dir.name}  L={L} H={H} step={S}  "
          f"参数量 {model.n_params() / 1e6:.3f} M  设备 {device}")

    res = predict_trajectory(
        model, normalizer, feat, L, H, S, device=device,
        horizon=args.horizon, batch_size=args.batch_size,
        stride=args.stride, max_windows=args.windows,
    )

    full_pred, full_true = res["full_pred"], res["full_true"]
    mae, mse, rmse, mape, mspe = metric(full_pred, full_true)
    pos = full_pred[..., :3] - full_true[..., :3]
    dist = np.linalg.norm(pos, axis=-1)
    h_used = int(args.horizon or H)

    summary = {
        "data_file": str(data_file),
        "checkpoint": str(ckpt_dir / args.ckpt_name),
        "n_windows": int(full_pred.shape[0]),
        "horizon": H,
        "display_horizon": h_used,
        "data_step": S,
        "stride_raw_points": int(args.stride),
        "mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "mspe": mspe,
        "position_mae_m": float(np.mean(np.abs(pos))),
        "position_distance_err_mean_m": float(dist.mean()),
        "position_distance_err_p90_m": float(np.percentile(dist, 90)),
        f"horizon{h_used}_position_mae_m": float(
            np.mean(np.abs(full_pred[:, h_used - 1, :3] - full_true[:, h_used - 1, :3]))),
        "per_dim": per_dim_error(full_pred.reshape(-1, TGT_DIM), full_true.reshape(-1, TGT_DIM)),
        "mae_by_horizon": error_by_horizon(full_pred, full_true).tolist(),
    }

    print("=" * 78)
    print(f"[结果] {full_pred.shape[0]:,} 个窗口 × {H} 步")
    print(f"  MAE {mae:.6f}   RMSE {rmse:.6f}")
    print(f"  位置 MAE                        : {summary['position_mae_m']:.3f} m")
    print(f"  空间距离误差 均值 / P90          : "
          f"{summary['position_distance_err_mean_m']:.3f} / "
          f"{summary['position_distance_err_p90_m']:.3f} m")
    print(f"  第 {h_used} 步位置 MAE                : "
          f"{summary[f'horizon{h_used}_position_mae_m']:.3f} m")
    print("=" * 78)

    np.savez(out_dir / "prediction.npz",
             full_pred=full_pred, full_true=full_true,
             values=res["values"], truths=res["truths"],
             timestamps=res["timestamps"], points_idx=res["points_idx"])
    dump_json(summary, out_dir / "metrics.json")

    # ---- 出图 ----
    L_ = L
    hist_end = int(res["start_index"][0]) + L_ * S
    history_abs = feat[: hist_end: S, 4:10]
    segments = [{
        "t_fut": (res["points_idx"] - res["points_idx"][0]) / S + len(history_abs),
        "true": res["truths"],
        "pred": res["values"],
    }]
    plot_prediction_timeline(history_abs, segments, out_dir / "timeline.png",
                             title=f"{data_file.stem} - rolling forecast, step {h_used}")
    plot_trajectory_3d(history_abs, res["truths"], res["values"],
                       out_dir / "trajectory_3d.png",
                       title=f"{data_file.stem} - 3D trajectory (step {h_used})")
    plot_sample_2d(history_abs, full_true[0], full_pred[0],
                   out_dir / "first_window_2d.png",
                   title=f"{data_file.stem} - window 1, all {H} steps")
    plot_horizon_error(error_by_horizon(full_pred, full_true), out_dir / "error_vs_horizon.png")

    print(f"[完成] 结果已保存到: {out_dir}")


if __name__ == "__main__":
    main()
