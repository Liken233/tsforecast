# Contributing

Thanks for your interest in `tsforecast`! 欢迎贡献。

## 开发环境

```bash
git clone https://github.com/Liken233/tsforecast.git
cd tsforecast
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## 提交前检查

```bash
ruff check src tests examples
pytest -q
```

`test_gui.py` 在没有显示环境时会自动跳过（CI 上无需 xvfb）。
如果想在无头机器上跑它，可以装 `xvfb` 并用 `xvfb-run pytest`。

## 代码约定

- 行宽 110，`ruff` 检查 `E/F/W/I`。
- 公开函数写中文 docstring，说明**参数形状与单位**（本项目所有张量/数组都有明确的维度约定）。
- 数值逻辑改动请**补测试**，尤其是 `data.py` 的窗口构造与归一化——那是整个包的口径来源。
- 不要提交数据集、训练权重、`checkpoint/`、`results/`（`.gitignore` 已覆盖）。

## 数据格式扩展

如果你的数据不是默认的 10 列布局，**优先考虑**先改 `src/tsforecast/data.py` 顶部的常量：

```python
N_COLS    = 10
ENC_SLICE = slice(1, 10)
TGT_SLICE = slice(4, 10)
ENC_DIM, TGT_DIM = 9, 6
```

如果想让列布局变成**可配置项**（而不是硬编码常量），欢迎提 PR：把 `data.py` 里对
`ENC_SLICE` / `TGT_SLICE` 的引用改为从 `cfg["data"]` 读取，并保留默认值以兼容现有 checkpoint。

## 报告问题

请附上：

- 完整的报错堆栈
- 你的数据结构（列数、`shape`、格式）与 `data_root` 布局
- `tsforecast-train --help` 中你用到的参数
- 环境：`python -c "import torch, sys; print(sys.version, torch.__version__)"`
