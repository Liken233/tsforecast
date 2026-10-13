# -*- coding: utf-8 -*-
"""pytest 公共夹具：合成数据集 + 小配置。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(ROOT / "src"))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def synth():
    """examples/make_synthetic_data.py 模块（顺便测这个示例脚本本身）。"""
    return _load_module(EXAMPLES / "make_synthetic_data.py", "synth_data")


@pytest.fixture(scope="session")
def data_root(tmp_path_factory, synth):
    """生成含 .mat/.csv/.npy 三种格式、两个子目录的数据集。"""
    root = tmp_path_factory.mktemp("dataset")
    for i, (group, mode, fmt) in enumerate([
        ("flightA", "glide", "mat"),
        ("flightA", "climb", "csv"),
        ("flightB", "weave", "npy"),
        ("flightB", "glide", "csv"),
    ]):
        arr = synth.generate_flight(n_points=6000, dt=0.5, seed=100 + i, mode=mode)
        synth.save_matrix(arr, root / group / f"{i + 1:04d}.{fmt}", fmt)
    return root


@pytest.fixture
def cfg(data_root):
    """小规模、CPU 友好的配置。"""
    return {
        "data": {
            "data_root": str(data_root),
            "datasets": [],
            "file_types": [".mat", ".csv", ".npy"],
            "mat_key": "dataMatrix",
            "csv_orientation": "auto",
            "csv_header": "auto",
            "input_len": 40,
            "pred_len": 20,
            "data_step": 50,
            "samples_per_file": 16,
            "cache_size": 2,
            "batch_size": 16,
            "num_workers": 0,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "split_by": "file",
            "seed": 42,
        },
        "model": {"d_model": 32, "num_heads": 4, "layer_num": 1, "dropout": 0.0},
        "train": {"epochs": 1, "learning_rate": 1e-3, "weight_decay": 0.0,
                  "grad_clip": 1.0, "use_amp": False},
        "hardware": {"use_gpu": False, "gpu": 0, "use_multi_gpu": False},
    }


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory, data_root):
    """
    造一个极小的训练产物目录（config.json + norm.json + checkpoint.pth），
    供 GUI / 推理 / 装配类测试使用，避免每次都真跑一遍训练。

    返回 dict(model_root, ckpt, cfg)：model_root 下只有一个子目录 ckpt。
    """
    import json

    import torch

    from tsforecast.data import build_loaders
    from tsforecast.models import Seq2SeqTrajectoryNet

    base = tmp_path_factory.mktemp("tinyck")
    model_root = base / "models"
    ckpt = model_root / "tiny_delta"
    ckpt.mkdir(parents=True)

    cfg_dict = {
        "data": {
            "data_root": str(data_root), "datasets": [],
            "file_types": [".mat", ".csv", ".npy"],
            "mat_key": "dataMatrix", "csv_orientation": "auto", "csv_header": "auto",
            "input_len": 40, "pred_len": 20, "data_step": 50,
            "samples_per_file": 8, "cache_size": 2, "batch_size": 8, "num_workers": 0,
            "val_ratio": 0.25, "test_ratio": 0.25, "split_by": "file", "seed": 42,
        },
        "model": {"d_model": 32, "num_heads": 4, "layer_num": 1, "dropout": 0.0},
        "train": {"checkpoint_root": str(model_root), "result_root": str(base / "results")},
        "hardware": {"use_gpu": False},
    }
    built = build_loaders(cfg_dict, verbose=False)
    built["normalizer"].save(str(ckpt / "norm.json"))
    model = Seq2SeqTrajectoryNet(input_dim=9, d_model=32, num_heads=4, pred_dim=6,
                                 pred_len=20, layer_num=1, dropout=0.0)
    torch.save(model.state_dict(), ckpt / "checkpoint.pth")
    (ckpt / "config.json").write_text(json.dumps(cfg_dict, ensure_ascii=False), encoding="utf-8")
    return {"model_root": model_root, "ckpt": ckpt, "cfg": cfg_dict}


@pytest.fixture
def rng():
    return np.random.default_rng(0)
