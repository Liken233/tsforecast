# -*- coding: utf-8 -*-
"""端到端冒烟测试：训练 / 评估 / 推理 三条 CLI + 程序化 API。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tsforecast.data import DeltaNormalizer, build_loaders, load_matrix
from tsforecast.inference import load_model_bundle, predict_trajectory
from tsforecast.models import Seq2SeqTrajectoryNet
from tsforecast.utils import error_by_horizon, metric, resolve_data_root

SMALL = ["--epochs", "1", "--d_model", "32", "--layer_num", "1", "--num_heads", "4",
         "--data_step", "50", "--samples_per_file", "12", "--batch_size", "8",
         "--no_amp", "--cpu"]


@pytest.fixture(scope="module")
def trained(tmp_path_factory, data_root):
    """跑一次真实 CLI 训练，供后续评估/推理用例复用。"""
    from tsforecast.train import main as train_main

    out = tmp_path_factory.mktemp("run")
    ckpt_root, res_root = out / "checkpoint", out / "results"
    train_main(["--exp_name", "smoke",
                "--data_root", str(data_root),
                "--checkpoint_root", str(ckpt_root),
                "--result_root", str(res_root)] + SMALL)
    return {"ckpt": ckpt_root / "smoke", "res": res_root / "smoke", "root": out,
            "data_root": str(data_root)}


def test_cli_train_产物齐全(trained):
    ck = trained["ckpt"]
    for f in ("checkpoint.pth", "config.json", "norm.json", "split.json"):
        assert (ck / f).exists(), f"缺少训练产物 {f}"
    for f in ("metrics.json", "history.json", "pred.npy", "true.npy",
              "loss_curve.png", "error_vs_horizon.png"):
        assert (trained["res"] / f).exists(), f"缺少训练图/指标 {f}"


def test_split_manifest_存相对路径(trained):
    man = json.loads((trained["ckpt"] / "split.json").read_text(encoding="utf-8"))
    for k in ("train", "val", "test"):
        for rel in man["splits"][k]:
            assert not Path(rel).is_absolute()
            assert "\\" not in rel, "应统一用 POSIX 风格分隔符"


def test_cli_evaluate(trained, tmp_path):
    from tsforecast.evaluate import main as eval_main

    out = tmp_path / "eval"
    eval_main(["--checkpoint_dir", str(trained["ckpt"]), "--split", "test",
               "--out_dir", str(out), "--batch_size", "8", "--num_workers", "0", "--cpu"])
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert m["position_mae_m"] >= 0
    assert set(m["per_dim"]) == {"x", "y", "z", "v", "pitch", "yaw"}
    assert len(m["mae_by_horizon"]) > 0
    for f in ("error_vs_horizon.png", "error_distribution.png", "sample_grid.png", "pred.npy"):
        assert (out / f).exists()


def test_cli_predict(trained, tmp_path):
    from tsforecast.predict import main as predict_main

    out = tmp_path / "pred"
    predict_main(["--checkpoint_dir", str(trained["ckpt"]),
                  "--data_root", trained["data_root"],
                  "--file_index", "0", "--windows", "20",
                  "--out_dir", str(out), "--cpu"])
    for f in ("prediction.npz", "timeline.png", "trajectory_3d.png",
              "first_window_2d.png", "metrics.json"):
        assert (out / f).exists()
    d = np.load(out / "prediction.npz")
    assert d["full_pred"].shape[-1] == 6
    assert d["full_pred"].shape == d["full_true"].shape


def test_评估支持_数据根目录覆盖(trained, tmp_path, data_root):
    """模拟换机器：把 config 里的 data_root 改成不存在的路径，再用 --data_root 救回。"""
    from tsforecast.evaluate import main as eval_main

    ckpt = tmp_path / "ckpt_copy"
    ckpt.mkdir()
    for f in ("checkpoint.pth", "norm.json", "split.json"):
        (ckpt / f).write_bytes((trained["ckpt"] / f).read_bytes())
    cfg_d = json.loads((trained["ckpt"] / "config.json").read_text(encoding="utf-8"))
    cfg_d["data"]["data_root"] = "D:/nonexistent/machine/dataset"
    (ckpt / "config.json").write_text(json.dumps(cfg_d, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        eval_main(["--checkpoint_dir", str(ckpt), "--split", "test",
                   "--out_dir", str(tmp_path / "e1"), "--cpu"])

    eval_main(["--checkpoint_dir", str(ckpt), "--split", "test",
               "--data_root", str(data_root),
               "--out_dir", str(tmp_path / "e2"), "--cpu"])
    assert (tmp_path / "e2" / "metrics.json").exists()


def test_程序化_API_全链路(cfg, tmp_path):
    built = build_loaders(cfg, verbose=False)
    nz = built["normalizer"]
    L, H = cfg["data"]["input_len"], cfg["data"]["pred_len"]

    model = Seq2SeqTrajectoryNet(input_dim=9, d_model=32, num_heads=4, pred_dim=6,
                                 pred_len=H, layer_num=1, dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lossf = torch.nn.MSELoss()

    first = last_loss = None
    for _ in range(6):
        for en, de, tg, _l in built["loaders"]["train"]:
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(en, de), tg)
            loss.backward()
            opt.step()
            last_loss = float(loss)
        if first is None:
            first = last_loss
    assert np.isfinite(last_loss) and last_loss < first, "训练 6 步后损失应下降"

    # 落盘 → 装配 → 推理
    ck = tmp_path / "ck"
    ck.mkdir()
    torch.save(model.state_dict(), ck / "checkpoint.pth")
    nz.save(str(ck / "norm.json"))
    (ck / "config.json").write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    m2, nz2, cfg2 = load_model_bundle(str(ck), device="cpu")
    assert cfg2["data"]["pred_len"] == H

    feat = load_matrix(str(sorted(Path(cfg["data"]["data_root"]).rglob("*.mat"))[0]))
    res = predict_trajectory(m2, nz2, feat, L, H, cfg["data"]["data_step"],
                             device="cpu", max_windows=50)
    assert res["full_pred"].shape[1:] == (H, 6)
    assert res["full_pred"].shape == res["full_true"].shape
    assert res["points_idx"].ndim == 1 and len(res["points_idx"]) == len(res["full_pred"])
    assert res["last_abs"].shape[-1] == 6

    mae, mse, rmse, mape, mspe = metric(res["full_pred"], res["full_true"])
    assert np.isfinite([mae, mse, rmse, mape, mspe]).all()
    assert error_by_horizon(res["full_pred"], res["full_true"]).shape == (H, 6)


def test_归一化器随_checkpoint_复用(cfg, tmp_path):
    built = build_loaders(cfg, verbose=False)
    nz = built["normalizer"]
    p = tmp_path / "norm.json"
    nz.save(str(p))
    back = DeltaNormalizer.load(str(p))
    x = np.random.default_rng(0).normal(size=(3, 5, 6))
    assert np.allclose(nz.denorm_tg(x), back.denorm_tg(x))


def test_resolve_data_root_报错可读(tmp_path):
    with pytest.raises(FileNotFoundError, match="数据目录不存在"):
        resolve_data_root(str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="未指定数据目录"):
        resolve_data_root(None)


def test_包内置配置存在且合法():
    from tsforecast.utils import DEFAULT_CONFIG

    assert DEFAULT_CONFIG.exists(), "包内置 default.json 丢失（检查 package-data 配置）"
    d = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    for key in ("data", "model", "train", "hardware"):
        assert key in d
