#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成演示用的**合成飞行器状态序列**，供快速上手与单元测试使用。

生成的每条序列是 10 列矩阵 (T, 10)，列布局与 `tsforecast.data` 的约定一致：

    col 0 : t (s)          col 1-3 : N_x / N_y / N_z（控制/过载指令）
    col 4-6 : x / y / z (m)            col 7 : v (m/s)
    col 8 : pitch (rad)                col 9 : yaw (rad)

用法:
    python examples/make_synthetic_data.py                        # 默认写到 ./data/dataset
    python examples/make_synthetic_data.py --out ./data/dataset --n-files 12
    python examples/make_synthetic_data.py --formats mat csv npy  # 三种格式都生成
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def generate_flight(n_points: int = 18000, dt: float = 0.5, seed: int = 0,
                    x0: float = 150_000.0, z0: float = 11_000.0,
                    v0: float = 320.0, mode: str = "glide") -> np.ndarray:
    """
    生成一条合成飞行状态序列，返回 (n_points, 10) float64。

    mode:
        ``glide``  滑翔下降段：高度单调下降、速度先增后减
        ``climb``  爬升段：高度上升、速度递减
        ``weave``  机动段：在滑翔基础上叠加侧向蛇行
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_points, dtype=np.float64) * dt
    s = t / max(t[-1], 1e-9)                     # 归一化进程 0→1

    if mode == "climb":
        z = z0 + 6000.0 * np.sin(np.pi * s * 0.8)
        v = v0 - 90.0 * s
    else:
        z = z0 - (z0 - 1500.0) * s ** 0.85
        v = v0 + 40.0 * np.sin(np.pi * s) - 30.0 * s

    x = x0 - (x0 - 12_000.0) * s + 600.0 * np.sin(2.2 * np.pi * s)
    y = 4_000.0 * np.sin(1.1 * np.pi * s) + 900.0 * np.sin(5.0 * np.pi * s)
    if mode == "weave":
        y = y + 2_500.0 * np.sin(7.0 * np.pi * s)

    # 加一点过程噪声，模拟测量/积分误差
    x = x + rng.normal(0.0, 6.0, n_points)
    y = y + rng.normal(0.0, 4.0, n_points)
    z = z + rng.normal(0.0, 3.0, n_points)
    v = v + rng.normal(0.0, 1.5, n_points)

    pitch = np.arctan2(np.gradient(-z, t), np.maximum(v, 1e-3))
    yaw = np.arctan2(np.gradient(y, t), np.maximum(np.gradient(x, t), -1e6))

    # 控制/过载指令：与姿态角速率同量级，作为额外的输入特征
    n_x = np.gradient(v, t) / 9.81
    n_y = np.gradient(yaw, t) * np.maximum(v, 1.0) / 9.81
    n_z = np.gradient(pitch, t) * np.maximum(v, 1.0) / 9.81

    return np.stack([t, n_x, n_y, n_z, x, y, z, v, pitch, yaw], axis=1).astype(np.float64)


def save_matrix(arr: np.ndarray, path: Path, fmt: str, mat_key: str = "dataMatrix") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        np.savetxt(path, arr, delimiter=",", fmt="%.6f",
                   header=",".join(["t", "N_x", "N_y", "N_z", "x", "y", "z", "v", "pitch", "yaw"]),
                   comments="")
    elif fmt == "npy":
        np.save(path, arr)
    elif fmt == "mat":
        import scipy.io as sio
        # 写 (10, T)，与常见 mat 导出习惯一致（读取端会自动转置）
        sio.savemat(path, {mat_key: arr.T})
    else:
        raise ValueError(f"未知格式: {fmt}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="生成合成飞行器状态序列")
    ap.add_argument("--out", type=str, default="./data/dataset",
                    help="输出根目录（其下按子目录分组，模拟多批次数据）")
    ap.add_argument("--n-files", type=int, default=8, help="生成文件数（自动分到 2 个子目录）")
    ap.add_argument("--n-points", type=int, default=18000, help="每条序列采样点数")
    ap.add_argument("--dt", type=float, default=0.5, help="采样间隔（秒）")
    ap.add_argument("--formats", nargs="+", default=["mat", "csv"],
                    choices=["mat", "csv", "npy"], help="输出格式（可多选）")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)

    root = Path(args.out)
    groups = ["flightA", "flightB"]
    modes = ["glide", "climb", "weave"]

    made = []
    for i in range(args.n_files):
        g = groups[i % len(groups)]
        arr = generate_flight(n_points=args.n_points, dt=args.dt,
                              seed=args.seed + i, mode=modes[i % len(modes)])
        for fmt in args.formats:
            name = f"{i + 1:04d}.{fmt}"
            save_matrix(arr, root / g / name, fmt)
            made.append(str(root / g / name))

    print("=" * 72)
    print(f"[完成] 生成 {len(made)} 个文件，序列长度 {args.n_points}，采样间隔 {args.dt}s")
    print(f"[目录] {root.resolve()}")
    for p in made[:6]:
        print("       ", Path(p).relative_to(root).as_posix())
    if len(made) > 6:
        print(f"        ... 其余 {len(made) - 6} 个")
    print("=" * 72)
    print("下一步:  tsforecast-train --data_root", root.as_posix())


if __name__ == "__main__":
    main()
