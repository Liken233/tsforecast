# -*- coding: utf-8 -*-
"""数据层测试：格式读取 / 目录发现 / Δ 窗口构造 / 归一化 / Dataset 与 Sampler。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tsforecast.data import (
    DeltaNormalizer, ENC_DIM, N_COLS, TGT_DIM,
    build_datasets, build_loaders, build_starts, describe_datasets,
    discover_datasets, load_csv, load_mat, load_matrix, load_npy,
    make_window, probe_length, TrajectoryWindowDataset, FileGroupedSampler,
)

ALL_EXTS = (".mat", ".csv", ".npy")


# ---------------------------------------------------------------- 格式读取
def test_load_三种格式形状一致(data_root, synth):
    files = sorted(data_root.rglob("0001.mat")) + sorted(data_root.rglob("0002.csv")) \
        + sorted(data_root.rglob("0003.npy"))
    assert len(files) == 3
    arrs = [load_matrix(str(f)) for f in files]
    for a in arrs:
        assert a.ndim == 2 and a.shape[1] == N_COLS
    assert arrs[0].shape == arrs[1].shape == arrs[2].shape


def test_mat_变量名回退(tmp_path, synth):
    """未提供 dataMatrix 时应自动回退到第一个二维变量。"""
    import scipy.io as sio
    p = tmp_path / "other.mat"
    sio.savemat(p, {"someMatrix": synth.generate_flight(2000, seed=7).T})
    got = load_matrix(str(p))
    assert got.shape == (2000, N_COLS)


def test_csv_表头与转置自适应(tmp_path, synth):
    arr = synth.generate_flight(1500, seed=3)
    p1 = tmp_path / "with_header.csv"
    np.savetxt(p1, arr, delimiter=",", header="a,b,c,d,e,f,g,h,i,j", comments="")
    p2 = tmp_path / "transposed.csv"
    np.savetxt(p2, arr.T, delimiter=",")
    assert load_csv(str(p1)).shape == (1500, N_COLS)
    assert load_csv(str(p2)).shape == (1500, N_COLS)


def test_npy_转置自适应(tmp_path, synth):
    arr = synth.generate_flight(1200, seed=11)
    np.save(tmp_path / "t.npy", arr)
    np.save(tmp_path / "c.npy", arr.T)
    assert load_matrix(str(tmp_path / "t.npy")).shape == (1200, N_COLS)
    assert load_matrix(str(tmp_path / "c.npy")).shape == (1200, N_COLS)


def test_不支持格式报错(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("1,2,3\n")
    with pytest.raises(ValueError):
        load_matrix(str(p))


def test_probe_length_与实际一致(data_root):
    for f in sorted(data_root.rglob("*"))[:4]:
        if f.suffix.lower() in ALL_EXTS:
            assert probe_length(str(f)) == load_matrix(str(f)).shape[0]


# ---------------------------------------------------------------- 目录发现
def test_discover_全部子目录(data_root):
    m = discover_datasets(str(data_root))
    assert set(m.keys()) == {"flightA", "flightB"}
    assert sum(len(v) for v in m.values()) == 4


def test_discover_按目录名与通配符(data_root):
    assert list(discover_datasets(str(data_root), ["flightA"]).keys()) == ["flightA"]
    assert set(discover_datasets(str(data_root), ["flight*"]).keys()) == {"flightA", "flightB"}
    assert set(discover_datasets(str(data_root), ["flightA", "flightB"]).keys()) == {"flightA", "flightB"}


def test_discover_不匹配时报错并提示(data_root):
    with pytest.raises(FileNotFoundError, match="未匹配到子目录"):
        discover_datasets(str(data_root), ["no_such_group"])


def test_discover_根目录不存在(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_datasets(str(tmp_path / "nope"))


def test_describe_datasets(data_root):
    txt = describe_datasets(discover_datasets(str(data_root)))
    assert "flightA" in txt and "flightB" in txt


# ---------------------------------------------------------------- Δ 窗口
def test_make_window_增量语义(synth):
    L, H, S = 10, 5, 3
    feat = synth.generate_flight(400, dt=0.5, seed=5)
    st = 20
    en, de, tg, last = make_window(feat, st, L, H, S)

    assert en.shape == (L, ENC_DIM)
    assert de.shape == (H, TGT_DIM)
    assert tg.shape == (H, TGT_DIM)
    assert last.shape == (TGT_DIM,)

    last_idx = st + (L - 1) * S
    assert np.allclose(last, feat[last_idx, 4:10])
    # 编码器增量：末点应为 0
    assert np.allclose(en[-1], 0.0, atol=1e-9)
    # 监督目标 = 未来绝对状态 − 窗口末状态
    tg_abs = feat[st + L * S: st + (L + H) * S: S, 4:10]
    assert np.allclose(tg, tg_abs - last, atol=1e-9)


def test_build_starts_越界保护():
    assert build_starts(100, 40, 20, 10, 8, np.random.default_rng(0)).size == 0
    st = build_starts(100_000, 40, 20, 5, 32, np.random.default_rng(0))
    assert st.size == 32
    assert st.min() >= 0 and st.max() + 60 * 5 <= 100_000


# ---------------------------------------------------------------- 归一化
def test_normalizer_往返一致():
    rng = np.random.default_rng(0)
    en = rng.normal(5.0, 2.0, (200, 10, ENC_DIM))
    tg = rng.normal(-3.0, 1.5, (200, 5, TGT_DIM))
    nz = DeltaNormalizer().fit(en, tg)

    # 全局（展平所有位置与样本）均值应恰好为 0、总体标准差应恰好为 1
    flat = nz.norm_en(en).reshape(-1, ENC_DIM)
    assert np.allclose(flat.mean(0), 0.0, atol=1e-10)
    assert np.allclose(flat.std(0), 1.0, atol=1e-10)

    assert np.allclose(nz.denorm_tg(nz.norm_tg(tg)), tg, atol=1e-9)

    last = rng.normal(100.0, 10.0, (200, TGT_DIM))
    abs_true = tg + last[:, None, :]
    assert np.allclose(nz.to_absolute(nz.norm_tg(tg), last), abs_true, atol=1e-9)


def test_normalizer_落盘读回(tmp_path):
    rng = np.random.default_rng(1)
    nz = DeltaNormalizer().fit(rng.normal(size=(50, 10, ENC_DIM)),
                              rng.normal(size=(50, 5, TGT_DIM)))
    p = tmp_path / "norm.json"
    nz.save(str(p))
    back = DeltaNormalizer.load(str(p))
    assert np.allclose(nz.mean_en, back.mean_en)
    assert np.allclose(nz.std_tg, back.std_tg)


def test_normalizer_零方差保护():
    nz = DeltaNormalizer().fit(np.zeros((10, 4, ENC_DIM)), np.zeros((10, 3, TGT_DIM)))
    assert np.all(np.isfinite(nz.std_en)) and np.all(nz.std_en > 0)


# ---------------------------------------------------------------- Dataset
def test_dataset_item_形状(cfg):
    built = build_datasets(cfg, verbose=False)
    ds = built["datasets"]["train"]
    en, de, tg, last = ds[0]
    L, H = cfg["data"]["input_len"], cfg["data"]["pred_len"]
    assert en.shape == (L, ENC_DIM)
    assert de.shape == (L, TGT_DIM)          # 解码器输入 = 前缀 H + 零占位 H
    assert tg.shape == (H, TGT_DIM)
    assert last.shape == (TGT_DIM,)
    assert torch.allclose(de[L - H:], torch.zeros(H, TGT_DIM))


def test_split_manifest_路径可迁移(cfg):
    """manifest 存相对路径：换 data_root 后仍能重建（换机器场景）。"""
    built = build_datasets(cfg, verbose=False)
    man = built["manifest"]
    for k in ("train", "val", "test"):
        for rel in man["splits"][k]:
            assert not rel.startswith("/") and ":" not in rel, f"{rel} 不是相对路径"

    rebuilt = build_datasets(cfg, split_manifest=man, verbose=False)
    a = [str(p) for p in built["datasets"]["test"].files]
    b = [str(p) for p in rebuilt["datasets"]["test"].files]
    assert a == b


def test_文件级划分无重叠(cfg):
    built = build_datasets(cfg, verbose=False)
    tr = {str(p) for p in built["datasets"]["train"].files}
    va = {str(p) for p in built["datasets"]["val"].files}
    te = {str(p) for p in built["datasets"]["test"].files}
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_归一化不受_val_test_组成影响(cfg):
    """归一化器只在训练集上拟合：交换 val/test 后统计量必须逐位相同。"""
    import copy
    b1 = build_datasets(cfg, verbose=False)
    man2 = copy.deepcopy(b1["manifest"])
    man2["splits"]["val"], man2["splits"]["test"] = (
        b1["manifest"]["splits"]["test"], b1["manifest"]["splits"]["val"])
    b2 = build_datasets(cfg, split_manifest=man2, verbose=False)
    assert np.allclose(b1["normalizer"].mean_en, b2["normalizer"].mean_en)
    assert np.allclose(b1["normalizer"].std_tg, b2["normalizer"].std_tg)


def test_file_grouped_sampler_覆盖全集(cfg):
    built = build_datasets(cfg, verbose=False)
    ds = built["datasets"]["train"]
    sampler = FileGroupedSampler(ds.file_span, seed=0)
    sampler.set_epoch(1)
    idx = list(sampler)
    assert sorted(idx) == list(range(len(ds)))
    assert len(sampler) == len(ds)


# ---------------------------------------------------------------- DataLoader
def test_build_loaders_返回结构(cfg):
    built = build_loaders(cfg, verbose=False)
    assert set(built["loaders"]) == {"train", "val", "test"}
    assert set(built["datasets"]) == {"train", "val", "test"}
    en, de, tg, last = next(iter(built["loaders"]["train"]))
    assert en.dim() == 3 and de.shape[0] == en.shape[0] == tg.shape[0] == last.shape[0]


def test_max_files_限制(cfg):
    built = build_datasets(cfg, verbose=False, max_files=1)
    for k in ("train", "val", "test"):
        assert len(built["datasets"][k].files) <= 1
