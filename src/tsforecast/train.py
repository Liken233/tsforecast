#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练入口：飞行器时序状态预测（增量 Δ 空间）。

环境变量或命令行二选一；装好包后推荐用控制台命令:
    tsforecast-train --help
    python -m tsforecast.train --help

用法示例:
    # 用包内置默认配置（数据取 ./data/dataset 下全部子目录）
    tsforecast-train

    # 指定数据根目录 + 只用其中两个子目录
    tsforecast-train --data_root ./data/dataset --datasets "flightA,flightB"

    # 快速试跑
    tsforecast-train --epochs 3 --samples_per_file 40 --max_files 4

    # 大模型配置（512 维 / 16 层 / batch 512）
    tsforecast-train --d_model 512 --layer_num 16 --batch_size 512

产物（相对路径落在当前工作目录下）:
    checkpoint/<setting>/   best 权重 checkpoint.pth、norm.json、config.json、split.json
    results/<setting>/      指标 metrics.json、loss_curve.png、pred.npy、true.npy
"""

from __future__ import annotations

import argparse
import copy
import datetime
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim


from tsforecast.data import build_loaders, ENC_DIM, TGT_DIM
from tsforecast.models import Seq2SeqTrajectoryNet                
from tsforecast.utils import (                                    
    DEFAULT_CONFIG, EarlyStopping, dump_json, error_by_horizon, load_json, metric,
    per_dim_error, plot_error_distribution, plot_horizon_error, plot_loss_curves,
    plot_sample_grid, plot_sample_2d, plot_trajectory_3d, resolve_device, resolve_path,
    set_seed, DIM_NAMES,
)


# ============================================================================
# 参数
# ============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="轨迹增量预测训练")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    p.add_argument("--exp_name", type=str, default=None, help="实验名（默认按时间戳生成）")

    g = p.add_argument_group("数据")
    g.add_argument("--data_root", type=str, default=None, help="数据根目录")
    g.add_argument("--datasets", type=str, default=None,
                   help="使用的子目录，逗号分隔；支持相对路径/目录名/通配符。留空=全部")
    g.add_argument("--input_len", type=int, default=None)
    g.add_argument("--pred_len", type=int, default=None)
    g.add_argument("--data_step", type=int, default=None)
    g.add_argument("--samples_per_file", type=int, default=None)
    g.add_argument("--max_files", type=int, default=None, help="每个子集最多用几个文件（调试用）")
    g.add_argument("--batch_size", type=int, default=None)
    g.add_argument("--num_workers", type=int, default=None)
    g.add_argument("--val_ratio", type=float, default=None)
    g.add_argument("--test_ratio", type=float, default=None)
    g.add_argument("--split_by", type=str, default=None, choices=["file", "dir"])
    g.add_argument("--seed", type=int, default=None)

    g = p.add_argument_group("模型")
    g.add_argument("--d_model", type=int, default=None)
    g.add_argument("--num_heads", type=int, default=None)
    g.add_argument("--layer_num", type=int, default=None)
    g.add_argument("--dropout", type=float, default=None)

    g = p.add_argument_group("训练")
    g.add_argument("--epochs", type=int, default=None)
    g.add_argument("--learning_rate", type=float, default=None)
    g.add_argument("--weight_decay", type=float, default=None)
    g.add_argument("--patience", type=int, default=None)
    g.add_argument("--grad_clip", type=float, default=None)
    g.add_argument("--checkpoint_root", type=str, default=None)
    g.add_argument("--result_root", type=str, default=None)
    g.add_argument("--no_amp", action="store_true", help="关闭混合精度")

    g = p.add_argument_group("硬件")
    g.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    g.add_argument("--gpu", type=int, default=None)
    return p.parse_args(argv)


def build_cfg(args) -> dict:
    cfg = load_json(args.config)

    def put(section, key, val):
        if val is not None:
            cfg.setdefault(section, {})[key] = val

    put("data", "data_root", args.data_root)
    put("data", "datasets",
        [s for s in args.datasets.replace(";", ",").split(",") if s.strip()]
        if args.datasets else None)
    for k in ("input_len", "pred_len", "data_step", "samples_per_file", "batch_size",
              "num_workers", "val_ratio", "test_ratio", "split_by", "seed"):
        put("data", k, getattr(args, k))

    for k in ("d_model", "num_heads", "layer_num", "dropout"):
        put("model", k, getattr(args, k))

    for k in ("epochs", "learning_rate", "weight_decay", "patience", "grad_clip",
              "checkpoint_root", "result_root"):
        put("train", k, getattr(args, k))
    if args.no_amp:
        cfg["train"]["use_amp"] = False

    put("hardware", "gpu", args.gpu)
    if args.cpu:
        cfg["hardware"]["use_gpu"] = False
        cfg["hardware"]["use_multi_gpu"] = False

    # 相对路径一律按当前工作目录解析（见 utils.resolve_path）
    cfg["data"]["data_root"] = resolve_path(cfg["data"]["data_root"])
    cfg["train"]["checkpoint_root"] = resolve_path(cfg["train"]["checkpoint_root"])
    cfg["train"]["result_root"] = resolve_path(cfg["train"]["result_root"])
    cfg["_max_files"] = args.max_files
    return cfg


# ============================================================================
# 前向 / 评估
# ============================================================================
def forward_loss(model, batch, criterion):
    en, de, tg, _last = batch
    pred = model(en, de)
    return pred, criterion(pred, tg)


@torch.no_grad()
def run_eval(model, loader, criterion, normalizer, device, use_amp, collect: bool = False):
    """
    返回 dict(loss=归一化 MSE, mae_xyz=物理空间位置 MAE(m), mae_all=全部维度 MAE)。
    collect=True 时额外返回绝对轨迹数组 pred_abs / true_abs。
    """
    model.eval()
    total, n = 0.0, 0
    preds, trues = [], []
    abs_err = []

    for batch in loader:
        batch = [t.to(device, non_blocking=True) for t in batch]
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred, loss = forward_loss(model, batch, criterion)
        total += loss.item()
        n += 1

        p = pred.float().cpu().numpy()
        t = batch[2].float().cpu().numpy()
        last = batch[3].float().cpu().numpy()[:, None, :]
        p_abs = normalizer.denorm_tg(p) + last
        t_abs = normalizer.denorm_tg(t) + last
        abs_err.append(np.abs(p_abs - t_abs))
        if collect:
            preds.append(p_abs)
            trues.append(t_abs)

    out = {
        "loss": total / max(n, 1),
        "mae_xyz": float(np.mean([e[..., :3].mean() for e in abs_err])),
        "mae_all": float(np.mean([e.mean() for e in abs_err])),
    }
    if collect:
        out["pred_abs"] = np.concatenate(preds, axis=0)
        out["true_abs"] = np.concatenate(trues, axis=0)
    return out


# ============================================================================
# 主流程
# ============================================================================
def main(argv=None):
    args = parse_args(argv)
    cfg = build_cfg(args)

    seed = int(cfg["data"]["seed"])
    set_seed(seed)
    device = resolve_device(cfg)

    # ---- 实验目录 ----
    ts = datetime.datetime.now().strftime("%m%d-%H%M")
    setting = args.exp_name or (
        f"{ts}_itr1_step{cfg['data']['data_step']}_lnum{cfg['model']['layer_num']}"
        f"_dmod{cfg['model']['d_model']}_bs{cfg['data']['batch_size']}"
    )
    ckpt_dir = Path(cfg["train"]["checkpoint_root"]) / setting
    res_dir = Path(cfg["train"]["result_root"]) / setting

    if not Path(cfg["data"]["data_root"]).is_dir():
        raise FileNotFoundError(
            f"数据根目录不存在: {cfg['data']['data_root']}\n"
            f"  用 --data_root 指定，或先跑 examples/make_synthetic_data.py 生成演示数据。"
        )

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"[实验] {setting}")
    print(f"[设备] {device}")
    print(f"[数据] root={cfg['data']['data_root']}  step={cfg['data']['data_step']}  "
          f"L={cfg['data']['input_len']}  H={cfg['data']['pred_len']}")
    print(f"[预测目标] 相对窗口末状态的增量 Δ (6 维: x,y,z,v,pitch,yaw)")
    print(f"[权重] {ckpt_dir}")
    print(f"[结果] {res_dir}")
    print("=" * 78)

    # ---- 数据 ----
    built = build_loaders(cfg, seed=seed, max_files=args.max_files)
    loaders = built["loaders"]
    normalizer = built["normalizer"]

    train_loader, val_loader, test_loader = loaders["train"], loaders["val"], loaders["test"]
    print(f"[数据] batch: train={len(train_loader)}  val={len(val_loader)}  test={len(test_loader)}")

    # ---- 模型 ----
    m = cfg["model"]
    model = Seq2SeqTrajectoryNet(
        input_dim=ENC_DIM, d_model=m["d_model"], num_heads=m["num_heads"],
        pred_dim=TGT_DIM, pred_len=cfg["data"]["pred_len"],
        layer_num=m["layer_num"], dropout=m["dropout"],
    )
    model = model.to(device)
    print(f"[模型] 参数量 {model.n_params() / 1e6:.3f} M")

    # ---- 优化器 / 调度器 ----
    tr = cfg["train"]
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=tr["learning_rate"],
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=tr["weight_decay"])
    epochs = int(tr["epochs"])
    train_steps = max(len(train_loader), 1)
    sched_name = tr.get("lr_schedule", "onecycle")
    if sched_name == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=tr["learning_rate"], steps_per_epoch=train_steps,
            epochs=epochs, pct_start=0.3)
    elif sched_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=tr["learning_rate"] * 0.01)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
            min_lr=tr["learning_rate"] * 0.01)

    use_amp = bool(tr.get("use_amp", True)) and device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    early_stopping = EarlyStopping(patience=int(tr["patience"]), verbose=True)

    # ---- 落盘实验配置 ----
    dump_json(cfg, ckpt_dir / "config.json")
    dump_json(built["manifest"], ckpt_dir / "split.json")
    normalizer.save(ckpt_dir / "norm.json")

    history = {"train_loss": [], "val_loss": [], "lr": [], "val_mae_xyz": [], "best_epoch": None}
    best_val = float("inf")

    # ========================================================================
    # 训练循环
    # ========================================================================
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        built["train_sampler"].set_epoch(epoch)
        model.train()
        ep_loss, nb = 0.0, 0
        ep_t0 = time.time()

        for it, batch in enumerate(train_loader):
            batch = [x.to(device, non_blocking=True) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                _pred, loss = forward_loss(model, batch, criterion)

            if use_amp:
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
                optimizer.step()

            if sched_name == "onecycle":
                scheduler.step()

            ep_loss += loss.item()
            nb += 1
            if (it + 1) % int(tr["print_freq"]) == 0:
                print(f"  Epoch {epoch}/{epochs} | batch {it + 1}/{train_steps} | "
                      f"loss {ep_loss / nb:.6f} | lr {optimizer.param_groups[0]['lr']:.3e}")

        if sched_name == "cosine":
            scheduler.step()

        train_loss = ep_loss / max(nb, 1)
        val = run_eval(model, val_loader, criterion, normalizer, device, use_amp)
        val_loss = val["loss"]
        if sched_name == "plateau":
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae_xyz"].append(val["mae_xyz"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        improved = early_stopping(val_loss, model, str(ckpt_dir))
        if improved:
            best_val = val_loss
            history["best_epoch"] = epoch

        print(f"Epoch {epoch}/{epochs} | train {train_loss:.6f} | val {val_loss:.6f} | "
              f"val位置MAE {val['mae_xyz']:.2f} m | 用时 {time.time() - ep_t0:.1f}s | "
              f"总 {(time.time() - t0) / 60:.1f} min")
        if early_stopping.early_stop:
            print(f"[早停] 触发于 epoch {epoch}")
            break

    # ========================================================================
    # 测试
    # ========================================================================
    best_path = ckpt_dir / "checkpoint.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
        print(f"[测试] 已加载最优权重 {best_path}")
    else:
        print("[测试] 未找到 checkpoint.pth，使用最后一轮权重")

    test = run_eval(model, test_loader, criterion, normalizer, device, use_amp, collect=True)
    pred_abs, true_abs = test["pred_abs"], test["true_abs"]

    mae, mse, rmse, mape, mspe = metric(pred_abs, true_abs)
    pos_mae = float(np.mean(np.abs(pred_abs[..., :3] - true_abs[..., :3])))
    final_err = float(np.mean(np.abs(pred_abs[:, -1, :3] - true_abs[:, -1, :3])))

    print("=" * 78)
    print(f"[测试结果] 形状 {pred_abs.shape}")
    print(f"  MAE  {mae:.6f}   MSE {mse:.6f}   RMSE {rmse:.6f}")
    print(f"  位置(xyz) 平均绝对误差 : {pos_mae:.3f} m")
    print(f"  末步(xyz) 平均绝对误差 : {final_err:.3f} m")
    print("=" * 78)

    metrics = {
        "setting": setting,
        "test_loss_normalized_mse": test["loss"],
        "mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "mspe": mspe,
        "position_mae_m": pos_mae,
        "last_step_position_mae_m": final_err,
        "per_dim": per_dim_error(pred_abs, true_abs),
        "best_val_loss": best_val,
        "pred_shape": list(pred_abs.shape),
        "epochs_run": len(history["train_loss"]),
    }
    dump_json(metrics, res_dir / "metrics.json")
    dump_json(history, res_dir / "history.json")
    np.save(res_dir / "pred.npy", pred_abs)
    np.save(res_dir / "true.npy", true_abs)
    np.save(res_dir / "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))

    # ---- 图 ----
    plot_loss_curves(history, res_dir / "loss_curve.png")
    plot_horizon_error(error_by_horizon(pred_abs, true_abs), res_dir / "error_vs_horizon.png")
    plot_error_distribution(pred_abs - true_abs, res_dir / "error_distribution.png")

    ds = built["datasets"]["test"]
    n_show = min(6, len(ds))
    samples = []
    for i in range(n_show):
        raw = ds.raw_absolute(i)
        samples.append({
            "history": raw["history_abs"],
            "true": raw["future_abs"],
            "pred": normalizer.denorm_tg(pred_abs[i:i + 1])[0] + raw["last_abs"],
            "name": Path(raw["file"]).stem,
        })
    plot_sample_grid(samples, res_dir / "sample_grid.png", title="Prediction vs ground truth (x)")
    s0 = samples[0]
    plot_sample_2d(s0["history"], s0["true"], s0["pred"],
                   res_dir / "sample_trajectory_2d.png", title=s0["name"])
    plot_trajectory_3d(s0["history"], s0["true"], s0["pred"],
                       res_dir / "sample_trajectory_3d.png", title=s0["name"])

    print(f"[完成] 权重: {ckpt_dir}")
    print(f"[完成] 结果: {res_dir}")
    print(f"[耗时] {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
