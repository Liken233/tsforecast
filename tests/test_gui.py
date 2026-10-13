# -*- coding: utf-8 -*-
"""GUI 冒烟测试：数据扫描 / 模型装配 / 预测 / 2D+3D 重绘。

无显示环境（CI、纯服务器）会自动跳过，不会让测试失败。
"""

from __future__ import annotations

import pytest


class _StubMessageBox:
    """把模态弹窗换成打印，避免无头测试被阻塞。"""

    @staticmethod
    def showinfo(*a, **k):
        print("[stub] showinfo:", a[1] if len(a) > 1 else a)

    @staticmethod
    def showerror(*a, **k):
        print("[stub] showerror:", a[1] if len(a) > 1 else a)

    @staticmethod
    def showwarning(*a, **k):
        print("[stub] showwarning:", a[1] if len(a) > 1 else a)


@pytest.fixture
def viewer(monkeypatch, data_root, tiny_checkpoint):
    tk = pytest.importorskip("tkinter", reason="需要 tkinter")
    try:
        root = tk.Tk()
    except Exception as exc:                      # pragma: no cover - 依赖运行环境
        pytest.skip(f"无可用显示环境: {exc}")

    viewer_mod = pytest.importorskip("tsforecast.gui.viewer", reason="需要 GUI 依赖")
    monkeypatch.setattr(viewer_mod, "messagebox", _StubMessageBox, raising=False)

    app = viewer_mod.SingleChartViewer(
        master=root,
        data_dir=str(data_root),
        model_dir=str(tiny_checkpoint["model_root"]),
        special_datasets_dir=str(tiny_checkpoint["model_root"]),
    )
    root.update()
    yield app, viewer_mod
    root.destroy()


def test_数据扫描识别全部条目(viewer):
    app, _ = viewer
    app.load_dataset_list()
    items = list(app.dataset_selector["values"])
    assert len(items) == 4
    assert any(i.startswith("flightA/") for i in items)
    assert any(i.startswith("flightB/") for i in items)


def test_模型装配只认三件套(viewer):
    app, _ = viewer
    app.load_all_models()
    assert len(app.models) == 1
    info = next(iter(app.models.values()))
    assert info["config"]["data"]["pred_len"] == 20
    assert info["normalizer"] is not None


def test_预测与重绘(viewer):
    app, _ = viewer
    app.load_dataset_list()
    app.load_all_models()

    did = sorted(app.dataset_selector["values"])[0]
    mid = list(app.models.keys())[0]
    app.dataset_var.set(did)
    app.model_var.set(app.models[mid]["name"])
    app.model_selected()

    # 窗口参数应从模型的 config.json 自动带出
    assert (app.input_len, app.output_len, app.pre_step) == (40, 20, 50)

    app.load_selected_dataset()
    res = app.predict_single_dataset(did, app.models[mid])
    assert res is not None
    assert res["values"].shape[1] == 6
    assert res["truths"].shape == res["values"].shape

    app.update_chart()
    app.master.update()
    assert app.view_mode == "2D"

    app.view_var.set("3D")
    app.view_mode_selected()
    app.update_chart()
    app.master.update()
    assert app.view_mode == "3D"

    app.show_all_features()
    app.master.update()


def test_窗口标题已通用化(viewer):
    app, _ = viewer
    assert app.master.title().startswith("Flight State")


def test_未知条目不崩溃(viewer):
    app, _ = viewer
    app.load_dataset_list()
    app.dataset_var.set("nonexistent/xxx")
    app.load_selected_dataset()          # 只弹提示，不应抛异常
