# tsforecast

**飞行器时序状态预测** — 用编码器-解码器 Transformer 预测多变量飞行状态序列（位置 / 速度 / 姿态角）的**增量 Δ**。

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-52%20passed-brightgreen)](#测试)
[![Code Style](https://img.shields.io/badge/style-ruff-000000)](https://github.com/astral-sh/ruff)

---

## 目录

- [这是什么](#这是什么)
- [核心思路：为什么预测增量 Δ](#核心思路为什么预测增量-δ)
- [特性](#特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [数据格式](#数据格式)
- [命令行参考](#命令行参考)
- [Python API](#python-api)
- [配置项](#配置项)
- [交互式可视化](#交互式可视化)
- [项目结构](#项目结构)
- [测试](#测试)
- [FAQ](#faq)
- [License](#license)

---

## 这是什么

给定一段飞行器的历史状态序列（位置、速度、姿态角），预测未来若干步的状态。适用于航迹外推、
状态估计前置预测、仿真环境中的目标运动模型等场景。

模型是 Informer 风格的 **encoder-decoder Transformer**：

```
                        ┌──────────────────────────────┐
  编码器输入 (L, 9)  ──▶ │ TokenEmbedding + 位置编码      │
  Δ 空间                │ Encoder × N (FullAttention)   │──┐
                        └──────────────────────────────┘  │
                                                          ▼
                        ┌──────────────────────────────┐  ┌──────────────────┐
  解码器输入 (L, 6)  ──▶ │ TokenEmbedding + 位置编码      │─▶│ Decoder × N      │
  [已知前缀 H | 零占位 H] │                              │  │ (自注意力+掩码   │
                        └──────────────────────────────┘  │  + 交叉注意力)   │
                                                          └────────┬─────────┘
                                                                   ▼ 取最后 H 步
                                                          ┌──────────────────┐
                                                          │ 逐步输出头        │──▶ (H, 6) Δ
                                                          └──────────────────┘
```

## 核心思路：为什么预测增量 Δ

直接回归状态绝对值有两个麻烦：数值范围大（位置可到 1e5 米量级）、不同场景起始点差异巨大。
本包改为预测**相对输入窗口末状态的增量**：

$$
\Delta_k \;=\; s_{t+k} - s_{\text{last}}, \qquad k = 1,\dots,H
$$

推理时按 $s_{t+k} = \Delta_k + s_{\text{last}}$ 还原为绝对序列。

这样做的收益：

- **与绝对量纲解耦**：网络只学"这段状态会怎么变"，换个起始点照样能用；
- **数值尺度友好**：增量动态范围远小于绝对量，配合 Δ 空间的全局归一化，训练更稳；
- **归一化统计量可复用**：mean / std 只在训练集上拟合并随 checkpoint 落盘，
  验证 / 测试 / 推理共用一套，杜绝数据泄漏与口径漂移。

## 特性

| | |
|---|---|
| **增量 Δ 预测** | 预测相对窗口末状态的位移，推理时加回参考状态即可 |
| **Δ 空间全局归一化** | 只在训练集拟合（流式累加，不占内存），随 checkpoint 落盘 |
| **多格式数据** | `.mat`（形状/变量名自适应）、`.csv`（方向/表头自适应）、`.npy` |
| **多目录选择** | `--datasets` 支持目录名 / 相对路径 / 通配符，逗号分隔；留空 = 全部 |
| **文件级划分** | `split_by=file` 按文件切 train/val/test，防止同源窗口泄漏 |
| **文件分组采样** | 同 batch 尽量来自同一文件，命中 LRU 整文件缓存，避免反复重读磁盘 |
| **可搬迁的划分清单** | `split.json` 存**相对路径**，整个数据集换个盘也不用改 |
| **混合精度 + 三种学习率调度** | OneCycle / CosineAnnealing / ReduceLROnPlateau，自动早停 |
| **开箱即用的可视化** | 训练曲线、逐维误差分布、误差-步长曲线、2D/3D 状态对比、交互式 Tkinter 查看器 |
| **GPU / CPU / 多卡** | 自动选择设备，`--cpu` / `--gpu N` / 多卡由配置控制 |

## 安装

```bash
git clone https://github.com/your-org/tsforecast.git
cd tsforecast
pip install -e .            # 开发模式安装
# 或
pip install -e ".[dev]"     # 额外装 pytest / ruff
```

依赖：`torch>=2.0`、`numpy>=1.24`、`scipy>=1.10`、`matplotlib>=3.7`。
交互式可视化额外需要 `tkinter`（Python 标准库自带；Debian/Ubuntu 上 `sudo apt install python3-tk`）。

安装后可用四个命令：

```bash
tsforecast-train     # 训练
tsforecast-eval      # 评估
tsforecast-predict   # 单条序列推理
tsforecast-gui       # 交互式可视化
```

没有装包也能跑（把 `src` 加进 `PYTHONPATH`）：

```bash
PYTHONPATH=src python -m tsforecast.train --help
```

## 快速开始

**1）生成一份合成飞行状态序列**（仓库不带数据，现场生成即可）：

```bash
python examples/make_synthetic_data.py --n-files 8 --n-points 18000 --formats mat csv npy
```

会在 `./data/dataset/flightA|flightB/` 下生成 10 列状态序列。

**2）训练**：

```bash
tsforecast-train --exp_name demo --epochs 30
```

**3）评估 / 推理 / 可视化**：

```bash
tsforecast-eval    --checkpoint_dir ./checkpoint/demo --split test
tsforecast-predict --checkpoint_dir ./checkpoint/demo --file_index 0 --windows 500
tsforecast-gui     --data_dir ./data/dataset --model_dir ./checkpoint
```

**4）或者用 Python API 一条龙**（见 [`examples/quickstart.py`](examples/quickstart.py)）：

```bash
python examples/quickstart.py --epochs 8
```

## 数据格式

每条样本是一段**单目标状态序列**，矩阵形状 `(T, 10)`（也接受 `(10, T)`，自动转置）：

| 列 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 含义 | `t` | `N_x` | `N_y` | `N_z` | `x` | `y` | `z` | `v` | `pitch` | `yaw` |
| 单位 | s | — | — | — | m | m | m | m/s | rad | rad |

- **编码器输入** = `col 1..9`（9 维：控制量 + 状态量）
- **预测目标** = `col 4..9`（6 维：位置 / 速度 / 姿态角）

> 列数不同怎么办？改 `src/tsforecast/data.py` 顶部的四个常量即可，其余代码不用动：
>
> ```python
> N_COLS    = 10
> ENC_SLICE = slice(1, 10)   # 编码器输入列
> TGT_SLICE = slice(4, 10)   # 预测目标列
> ENC_DIM, TGT_DIM = 9, 6
> ```

**目录布局**：`data_root` 下可有多级子目录，递归扫描，每个「直接含数据文件的目录」算一个数据集：

```
data/dataset/
├── flightA/
│   ├── 0001.mat
│   └── 0002.csv
└── flightB/
    └── 0003.npy
```

**各格式读取规则**：

| 格式 | 规则 |
|---|---|
| `.mat` | 变量名默认 `dataMatrix`；不存在时自动回退到第一个二维变量；形状自动转置 |
| `.csv` | 默认「行=时间步、列=特征」；也支持转置；有无表头自动识别；`,` 与 `;` 都兼容 |
| `.npy` | `np.load` 的二维数组，方向自动判定 |

**样本构造**（`L` = `input_len`，`H` = `pred_len`，`S` = `data_step`）：

```
索引:  st ─────────── st+(L-1)S ─── st+L·S ─────── st+(L+H)·S
       │← 编码器 L 步 →│← 预测 H 步 →│

en_delta = feat[st : st+L·S : S, 1:10] − feat[st+(L-1)S, 1:10]      (L, 9)
de_delta = feat[st+(L-H)·S : st+L·S : S, 4:10] − feat[st+(L-1)S, 4:10]  (H, 6)
tg_delta = feat[st+L·S : st+(L+H)·S : S, 4:10] − feat[st+(L-1)S, 4:10]  (H, 6)
last_abs = feat[st+(L-1)S, 4:10]                                    (6,)
```

解码器输入 = `concat([de_delta, zeros(H, 6)])`（长度 `L`），网络取输出**最后 H 步**。

## 命令行参考

### `tsforecast-train`

| 参数 | 默认 | 说明 |
|---|---|---|
| `--config` | 包内置 `configs/default.json` | 配置文件路径 |
| `--exp_name` | 时间戳+超参 | 实验名，决定产物目录 |
| `--data_root` | `./data/dataset` | 数据根目录 |
| `--datasets` | 空=全部 | 子目录名 / 相对路径 / 通配符，逗号分隔，如 `"flightA,flight*"` |
| `--input_len` `--pred_len` `--data_step` | 40 / 20 / 100 | 编码器步数 / 预测步数 / 采样间隔 |
| `--samples_per_file` | 300 | 每个文件抽多少窗口 |
| `--max_files` | None | 每个子集最多用几个文件（调试） |
| `--split_by` | `file` | `file` 或 `dir` |
| `--val_ratio` `--test_ratio` | 0.1 / 0.1 | 验证 / 测试文件占比 |
| `--d_model` `--num_heads` `--layer_num` `--dropout` | 256 / 8 / 6 / 0.1 | 网络规模 |
| `--epochs` `--learning_rate` `--weight_decay` `--patience` `--grad_clip` | 50 / 1e-4 / 1e-2 / 10 / 1.0 | 训练超参 |
| `--num_workers` | 0 | DataLoader 进程数（服务器建议 8） |
| `--no_amp` `--cpu` `--gpu N` | — | 关混合精度 / 强制 CPU / 指定 GPU |

### `tsforecast-eval`

| 参数 | 说明 |
|---|---|
| `--checkpoint_dir` | **必填**，训练产物目录 |
| `--split` | `train` / `val` / `test`，默认 `test` |
| `--data_root` | 覆盖 checkpoint 里记录的数据路径（换机器时用） |
| `--datasets` | 只评估其中几个子目录 |
| `--out_dir` | 输出目录（默认 `<result_root>/<setting>/eval`） |

### `tsforecast-predict`

| 参数 | 说明 |
|---|---|
| `--checkpoint_dir` | **必填** |
| `--data_file` | 直接指定数据文件 |
| `--data_root` `--datasets` `--file_index` | 或按目录+序号挑 |
| `--horizon` | 展示第几步的预测（默认 = `pred_len`） |
| `--stride` `--windows` | 滑窗步长 / 最多用几个窗口 |
| `--out_dir` | 输出目录（默认 `<result_root>/<setting>/predict`） |

### 产物

```
checkpoint/<setting>/   checkpoint.pth  config.json  norm.json  split.json
results/<setting>/      metrics.json  history.json  pred.npy  true.npy  metrics.npy
                        loss_curve.png  error_vs_horizon.png  error_distribution.png
                        sample_grid.png  sample_trajectory_2d.png  sample_trajectory_3d.png
results/<setting>/eval/    metrics.json + 误差图 + pred.npy/true.npy
results/<setting>/predict/ prediction.npz + timeline.png + trajectory_3d.png + metrics.json
```

`metrics.json` 主要字段：`mae` / `mse` / `rmse`、`position_mae_m`、
`position_distance_err_mean_m` / `_p90_m`、`last_step_position_mae_m`、
`per_dim`（逐维误差）、`mae_by_horizon`（误差随步长曲线）。

## Python API

```python
import torch
from tsforecast.data import build_loaders, ENC_DIM, TGT_DIM
from tsforecast.models import Seq2SeqTrajectoryNet
from tsforecast.inference import load_model_bundle, predict_trajectory
from tsforecast.utils import resolve_device

cfg = {
    "data":  {"data_root": "./data/dataset", "datasets": [], "input_len": 40,
              "pred_len": 20, "data_step": 50, "batch_size": 64, "num_workers": 0,
              "val_ratio": 0.2, "test_ratio": 0.2, "split_by": "file", "seed": 42},
    "model": {"d_model": 128, "num_heads": 4, "layer_num": 3, "dropout": 0.1},
    "hardware": {"use_gpu": True, "gpu": 0},
}

built = build_loaders(cfg, verbose=True)
loaders, normalizer = built["loaders"], built["normalizer"]

model = Seq2SeqTrajectoryNet(input_dim=ENC_DIM, pred_dim=TGT_DIM,
                             pred_len=cfg["data"]["pred_len"], **cfg["model"])

# ... 训练循环见 examples/quickstart.py ...

# 推理：Δ → 绝对
m, nz, _ = load_model_bundle("./checkpoint/demo", device="cpu")
res = predict_trajectory(m, nz, feat, input_len=40, pred_len=20, data_step=50,
                         device="cpu", max_windows=200)
print(res["full_pred"].shape)     # (n, 20, 6) 绝对状态预测
print(res["full_true"].shape)     # (n, 20, 6)
```

## 配置项

完整默认配置见 [`src/tsforecast/configs/default.json`](src/tsforecast/configs/default.json)。

| 段 | 键 | 说明 |
|---|---|---|
| `data` | `data_root` `datasets` `file_types` | 数据位置与筛选 |
| | `mat_key` `csv_orientation` `csv_header` | 各格式读取细节 |
| | `input_len` `pred_len` `data_step` `samples_per_file` | 窗口与采样 |
| | `cache_size` `batch_size` `num_workers` | 性能 |
| | `val_ratio` `test_ratio` `split_by` `seed` | 划分与复现 |
| `model` | `d_model` `num_heads` `layer_num` `dropout` | 网络规模 |
| `train` | `epochs` `learning_rate` `weight_decay` `lr_schedule` `grad_clip` `patience` | 优化 |
| | `use_amp` `print_freq` `checkpoint_root` `result_root` | 运行 |
| `hardware` | `use_gpu` `gpu` `use_multi_gpu` `devices` | 设备 |

**相对路径一律相对当前工作目录解析**，因此建议在项目根目录下执行命令。

## 交互式可视化

```bash
tsforecast-gui --data_dir ./data/dataset --model_dir ./checkpoint
```

左侧浏览数据（递归扫描 `.mat/.csv/.npy`），中间选模型，右侧联动显示：

- **2D 三视图**：位置 / 速度 / 姿态角的时间曲线，预测与真值叠加
- **单特征视图**：放大看某一维
- **3D 视图**：三维状态轨迹，含朝向指示

模型目录下每个子目录须同时具备 `config.json` + `norm.json` + `*.pth` 才会被识别，
网络结构与窗口参数（`input_len` / `pred_len` / `data_step`）全部从 `config.json` 自动读取。

> 无显示环境（服务器 / CI）请用 `tsforecast-predict` 出静态图。

## 项目结构

```
tsforecast/
├── pyproject.toml
├── LICENSE
├── README.md
├── src/tsforecast/
│   ├── __init__.py
│   ├── data.py           数据层：文件发现 / 读取 / Δ 窗口 / 归一化 / Dataset / Sampler
│   ├── models.py         编码器-解码器 Transformer
│   ├── utils.py          路径 / 种子 / 指标 / 早停 / 绘图
│   ├── inference.py      模型装配 + 整段滚动滑窗推理
│   ├── train.py          训练入口        → tsforecast-train
│   ├── evaluate.py       评估入口        → tsforecast-eval
│   ├── predict.py        推理入口        → tsforecast-predict
│   ├── configs/default.json
│   └── gui/viewer.py     交互式可视化    → tsforecast-gui
├── examples/
│   ├── make_synthetic_data.py   合成数据生成
│   └── quickstart.py            Python API 一条龙
└── tests/
    ├── conftest.py
    ├── test_data.py      格式读取 / 目录发现 / Δ 语义 / 归一化 / 划分
    ├── test_models.py    前向形状 / 因果掩码 / 梯度 / state_dict 键名
    ├── test_smoke.py     训练-评估-推理 三条 CLI + 程序化 API
    └── test_gui.py       GUI 冒烟（无显示环境自动跳过）
```

## 测试

```bash
pip install -e ".[dev]"
pytest -q
```

52 项用例覆盖：三种文件格式的读取与方向自适应、目录/通配符筛选、增量 Δ 的数学语义、
归一化往返与"只在训练集拟合"的不变量、文件级划分无重叠、划分清单可搬迁、
模型前向/因果性/梯度/权重键名稳定性、三条 CLI 端到端、GUI 扫描与重绘。

## FAQ

**Q：换了台机器，`tsforecast-eval` 报"数据目录不存在"？**
checkpoint 里记录的是训练时的路径。用 `--data_root` 指向本机数据根目录即可；
`split.json` 存的是相对路径，不用改。

**Q：我的数据不是 10 列 / 不是这 10 个物理量？**
改 `src/tsforecast/data.py` 顶部的 `N_COLS` / `ENC_SLICE` / `TGT_SLICE` / `ENC_DIM` / `TGT_DIM`，
另外 `utils.py` 的 `DIM_NAMES` 与绘图里的维度标签可按需同步。

**Q：显存不够？**
减小 `--batch_size`、`--d_model`、`--layer_num`，或开 `--no_amp` 之外的混合精度（默认已开）。

**Q：训练很慢，数据反复重读？**
把 `--num_workers` 调到 8 左右，并适当增大 `cache_size`（缓存整个文件的窗口）。

**Q：想换学习率调度？**
改配置里的 `train.lr_schedule`：`onecycle` / `cosine` / `plateau`。

## License

[MIT](LICENSE)
