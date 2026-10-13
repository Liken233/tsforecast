# -*- coding: utf-8 -*-
"""
统一数据层：文件发现 / mat+csv+npy 读取 / 增量(Δ)样本构造 / 全局归一化 / 数据集与采样器。

原始状态序列矩阵的列布局（10 列）:

    col 0 : 时间戳 t (s)
    col 1 : N_x        控制/过载指令
    col 2 : N_y
    col 3 : N_z
    col 4 : x          位置 (m)
    col 5 : y
    col 6 : z
    col 7 : v          速度 (m/s)
    col 8 : pitch      俯仰角 (rad)
    col 9 : yaw        偏航角 (rad)

- 编码器输入 = col 1..9  (ENC_DIM=9)：状态量 + 控制量，全部作为输入特征
- 预测目标   = col 4..9  (TGT_DIM=6)：位置 / 速度 / 姿态角

若你的数据列数不同，按下表改 `ENC_SLICE` / `TGT_SLICE` / `ENC_DIM` / `TGT_DIM` 即可，
其余代码无需改动。

支持的三种文件格式:
- .mat : 内部变量默认名为 ``dataMatrix``，形状 (10, T)（行=特征、列=时间）。
         若同名变量不存在，自动回退到第一个非系统的二维变量；读入后按形状自动转置成 (T, 10)。
- .csv : 逗号分隔，默认「行=时间步、列=特征」(T, 10)；也支持「行=特征」(10, T)。
         表头自动识别（首行存在非数字字段时视为表头跳过）。分号分隔同样兼容。
- .npy : 直接 ``np.load`` 的二维数组，(T, 10) 或 (10, T) 自动判定。

预测量的定义（本包的核心约定）:
    预测的不是状态绝对值，而是**相对“输入窗口末状态”的增量** Δ。
    设输入窗口最后一个采样点的绝对状态为 s_last（6 维: x, y, z, v, pitch, yaw），
    则未来第 k 步的监督目标为  Δ_k = s_{t+k} - s_last。
    推理时按  s_{t+k} = Δ_k + s_last  还原为绝对序列。
    这样做的好处：网络只学“状态增量”，与初始位置 / 绝对量纲解耦，跨场景泛化更好。
"""

from __future__ import annotations

import csv
import fnmatch
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

# ----------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------
N_COLS = 10
ENC_SLICE = slice(1, 10)      # 编码器输入列
TGT_SLICE = slice(4, 10)      # 预测目标列
ENC_DIM = 9
TGT_DIM = 6
DEFAULT_EXTS = (".mat", ".csv", ".npy")

COLUMN_NAMES = ["t", "N_x", "N_y", "N_z", "x", "y", "z", "v", "pitch", "yaw"]
UNIT_HINT = ["s", "-", "-", "-", "m", "m", "m", "m/s", "rad", "rad"]


# ----------------------------------------------------------------------------
# 文件发现
# ----------------------------------------------------------------------------
def _iter_data_files(root: Path, exts: Sequence[str]) -> Dict[str, List[Path]]:
    """递归扫描 root，返回 {相对子目录: [文件...]}，只保留直接含数据文件的目录。"""
    exts = tuple(e.lower() for e in exts)
    buckets: Dict[str, List[Path]] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            rel = p.parent.relative_to(root)
            key = "." if str(rel) == "." else rel.as_posix()
            buckets.setdefault(key, []).append(p)
    return buckets


def discover_datasets(
    data_root: str,
    datasets: Optional[Sequence[str]] = None,
    exts: Sequence[str] = DEFAULT_EXTS,
) -> "OrderedDictCompat":
    """
    发现 data_root 下所有「直接包含数据文件」的子目录。

    参数:
        data_root : 数据根目录，例如 ./data/dataset
        datasets  : 想使用的子目录。支持三种写法，可混用、逗号分隔:
                      - 相对路径   'campaign2024/flightA'
                      - 目录名     'flightA'
                      - 通配符     'campaign2024/*' 或 'flight*'
                    留空 / None / [] → 使用全部子目录（默认行为）。
        exts      : 参与匹配的扩展名。

    返回:
        dict {相对目录名: [文件路径...]}，按目录名排序。
    """
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"数据根目录不存在: {root}")

    buckets = _iter_data_files(root, exts)
    if not buckets:
        raise FileNotFoundError(
            f"在 {root} 下未找到任何 {exts} 文件（已递归搜索）"
        )

    keys = sorted(buckets.keys())

    # 未指定 → 全部
    if datasets is None:
        picked = keys
    else:
        if isinstance(datasets, str):
            datasets = [s for s in datasets.replace(";", ",").split(",")]
        patterns = [str(s).strip().strip("/\\") for s in datasets if str(s).strip()]
        if not patterns:
            picked = keys
        else:
            picked = []
            for pat in patterns:
                hit = [
                    k for k in keys
                    if k == pat
                    or Path(k).name == pat
                    or fnmatch.fnmatch(k, pat)
                    or fnmatch.fnmatch(Path(k).name, pat)
                ]
                if not hit:
                    raise FileNotFoundError(
                        f"未匹配到子目录: {pat!r}\n可选子目录:\n  "
                        + "\n  ".join(keys)
                    )
                for h in hit:
                    if h not in picked:
                        picked.append(h)

    result = {k: buckets[k] for k in keys if k in picked}
    return result


class OrderedDictCompat(dict):
    pass


def describe_datasets(dataset_map: Dict[str, List[Path]]) -> str:
    """生成人类可读的数据集清单。"""
    lines = [f"{'子目录':<44s}{'文件数':>8s}   样例"]
    total = 0
    for k, v in dataset_map.items():
        total += len(v)
        lines.append(f"{k:<44s}{len(v):>8d}   {v[0].name}")
    lines.append(f"{'合计':<44s}{total:>8d}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 单文件读取
# ----------------------------------------------------------------------------
def _orient_to_time_major(arr: np.ndarray, tag: str) -> np.ndarray:
    """把 (10, T) / (T, 10) 统一成 (T, 10)。"""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{tag}: 期望二维矩阵，实际 shape={arr.shape}")
    if arr.shape[1] == N_COLS:
        out = arr
    elif arr.shape[0] == N_COLS:
        out = arr.T
    else:
        raise ValueError(
            f"{tag}: 列数不匹配，需要 {N_COLS} 个特征，实际 shape={arr.shape}"
        )
    return np.ascontiguousarray(out)


def load_mat(path: str, mat_key: str = "dataMatrix") -> np.ndarray:
    """读取 .mat，返回 (T, 10)。"""
    import scipy.io as sio

    raw = sio.loadmat(str(path))
    keys = [k for k in raw.keys() if not k.startswith("__")]

    if mat_key in keys:
        data = raw[mat_key]
    else:
        cand = [k for k in keys if np.asarray(raw[k]).ndim == 2]
        if not cand:
            raise KeyError(
                f"{Path(path).name}: 未找到变量 {mat_key!r}，且没有可用的二维变量（实际: {keys}）"
            )
        mat_key = cand[0]

    return _orient_to_time_major(raw[mat_key], Path(path).name)


def _csv_has_header(path: str) -> bool:
    """首行只要出现任一非数字字段即判定为表头。"""
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cells = [c.strip() for c in line.replace(";", ",").split(",")]
            for c in cells:
                if c == "":
                    continue
                try:
                    float(c)
                except ValueError:
                    return True
            return False
    return False


def load_csv(
    path: str,
    orientation: str = "auto",
    header: str = "auto",
) -> np.ndarray:
    """读取 .csv，返回 (T, 10)。orientation: auto|col(行=时间)|row(行=特征)。"""
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        text = f.read()

    if header == "auto":
        skip = 1 if _csv_has_header(path) else 0
    else:
        skip = 1 if str(header).lower() in ("1", "true", "yes") else 0

    rows = []
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if i == 0 and skip:
            continue
        cells = [c for c in line.replace(";", ",").split(",")]
        try:
            rows.append([float(c) if c.strip() != "" else np.nan for c in cells])
        except ValueError:
            continue

    if not rows:
        raise ValueError(f"{Path(path).name}: CSV 未解析出任何数值行")
    arr = np.asarray(rows, dtype=np.float64)

    if orientation == "row":
        return _orient_to_time_major(arr.T, Path(path).name)
    if orientation == "col":
        return _orient_to_time_major(arr, Path(path).name)
    # auto
    if arr.shape[1] == N_COLS:
        return _orient_to_time_major(arr, Path(path).name)
    if arr.shape[0] == N_COLS:
        return _orient_to_time_major(arr.T, Path(path).name)
    raise ValueError(f"{Path(path).name}: CSV 形状 {arr.shape} 无法判定列数（需要 {N_COLS}）")


def load_npy(path: str) -> np.ndarray:
    """读取 .npy，返回 (T, 10)。(T, 10) 或 (10, T) 自动判定。"""
    arr = np.load(str(path), allow_pickle=False)
    if arr.ndim != 2:
        raise ValueError(f"{Path(path).name}: 期望二维数组，实际 shape={arr.shape}")
    return _orient_to_time_major(arr, Path(path).name)


def load_matrix(
    path: str,
    mat_key: str = "dataMatrix",
    csv_orientation: str = "auto",
    csv_header: str = "auto",
) -> np.ndarray:
    """统一入口：按扩展名分派，返回 (T, 10) float64。"""
    suffix = Path(path).suffix.lower()
    if suffix == ".mat":
        return load_mat(path, mat_key=mat_key)
    if suffix == ".csv":
        return load_csv(path, orientation=csv_orientation, header=csv_header)
    if suffix == ".npy":
        return load_npy(path)
    raise ValueError(f"不支持的文件类型: {suffix}（支持 .mat / .csv / .npy）")


# ----------------------------------------------------------------------------
# 增量(Δ)归一化
# ----------------------------------------------------------------------------
class DeltaNormalizer:
    """
    在「位移(Δ)空间」做全局标准化。

    统计量只在训练集上拟合（fit），随 checkpoint 一起落盘，
    验证/测试/推理/后续接入仿真环境时复用同一套参数，保证可复现。
    """

    def __init__(
        self,
        mean_en: Optional[np.ndarray] = None,
        std_en: Optional[np.ndarray] = None,
        mean_tg: Optional[np.ndarray] = None,
        std_tg: Optional[np.ndarray] = None,
    ):
        self.mean_en = np.zeros(ENC_DIM, np.float64) if mean_en is None else np.asarray(mean_en, np.float64)
        self.std_en = np.ones(ENC_DIM, np.float64) if std_en is None else np.asarray(std_en, np.float64)
        self.mean_tg = np.zeros(TGT_DIM, np.float64) if mean_tg is None else np.asarray(mean_tg, np.float64)
        self.std_tg = np.ones(TGT_DIM, np.float64) if std_tg is None else np.asarray(std_tg, np.float64)
        self._guard()

    # -- 统计量累计 -------------------------------------------------------
    @staticmethod
    def _acc(acc, arr: np.ndarray):
        arr = np.asarray(arr, np.float64)
        flat = arr.reshape(-1, arr.shape[-1])
        acc["n"] += flat.shape[0]
        acc["sum"] += flat.sum(0)
        acc["sq"] += np.square(flat).sum(0)

    @staticmethod
    def _finalize(acc, eps: float = 1e-12):
        n = max(acc["n"], 1)
        mean = acc["sum"] / n
        var = np.maximum(acc["sq"] / n - np.square(mean), 0.0)
        std = np.sqrt(var)
        std[std < eps] = 1.0
        return mean, std

    @staticmethod
    def _new_acc(dim: int):
        return {"n": 0, "sum": np.zeros(dim, np.float64), "sq": np.zeros(dim, np.float64)}

    def fit(self, en_deltas: np.ndarray, tg_deltas: np.ndarray) -> "DeltaNormalizer":
        """一次性拟合（传入 n×L×9 与 n×H×6 的原始增量数组）。"""
        a_en, a_tg = self._new_acc(ENC_DIM), self._new_acc(TGT_DIM)
        self._acc(a_en, en_deltas)
        self._acc(a_tg, tg_deltas)
        self.mean_en, self.std_en = self._finalize(a_en)
        self.mean_tg, self.std_tg = self._finalize(a_tg)
        self._guard()
        return self

    class _BatchAccumulator:
        """流式累计器（逐文件喂入，避免把全量数据堆在内存里）。"""

        def __init__(self):
            self.a_en = DeltaNormalizer._new_acc(ENC_DIM)
            self.a_tg = DeltaNormalizer._new_acc(TGT_DIM)

        def update(self, en_deltas, tg_deltas):
            DeltaNormalizer._acc(self.a_en, en_deltas)
            DeltaNormalizer._acc(self.a_tg, tg_deltas)

        def finalize(self) -> "DeltaNormalizer":
            norm = DeltaNormalizer()
            norm.mean_en, norm.std_en = DeltaNormalizer._finalize(self.a_en)
            norm.mean_tg, norm.std_tg = DeltaNormalizer._finalize(self.a_tg)
            norm._guard()
            return norm

    @staticmethod
    def accumulator() -> "DeltaNormalizer._BatchAccumulator":
        return DeltaNormalizer._BatchAccumulator()

    # -- 变换 -------------------------------------------------------------
    def _guard(self):
        for name in ("std_en", "std_tg"):
            s = getattr(self, name)
            s[~np.isfinite(s)] = 1.0
            s[s < 1e-12] = 1.0

    def norm_en(self, en_delta: np.ndarray) -> np.ndarray:
        return (en_delta - self.mean_en) / self.std_en

    def norm_tg(self, tg_delta: np.ndarray) -> np.ndarray:
        return (tg_delta - self.mean_tg) / self.std_tg

    def denorm_tg(self, tg_norm: np.ndarray) -> np.ndarray:
        """归一化增量 → 物理增量 (m, m, m, m/s, rad, rad)。"""
        return tg_norm * self.std_tg + self.mean_tg

    def to_absolute(self, tg_norm: np.ndarray, last_abs: np.ndarray) -> np.ndarray:
        """
        归一化增量 + 窗口末绝对状态 → 绝对序列。

        自动补齐广播维度，下面两种写法等价::

            to_absolute(p_norm, last)              # last: [B, 6]
            to_absolute(p_norm, last[:, None, :])  # last: [B, 1, 6]

        其中 p_norm 形如 [B, H, 6]（也可以是 [H, 6] / [6]）。
        """
        delta = self.denorm_tg(tg_norm)
        last = np.asarray(last_abs, np.float64)
        while last.ndim < delta.ndim:
            last = np.expand_dims(last, -2)
        return delta + last

    # -- 落盘 -------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "space": "delta",
            "mean_en": self.mean_en.tolist(),
            "std_en": self.std_en.tolist(),
            "mean_tg": self.mean_tg.tolist(),
            "std_tg": self.std_tg.tolist(),
            "encoder_columns": COLUMN_NAMES[ENC_SLICE],
            "target_columns": COLUMN_NAMES[TGT_SLICE],
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "DeltaNormalizer":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["mean_en"], d["std_en"], d["mean_tg"], d["std_tg"])


def probe_length(path: str, mat_key: str = "dataMatrix", csv_orientation: str = "auto",
                 csv_header: str = "auto") -> int:
    """只取序列长度，避免为探测而整表载入（.mat 走 whosmat，.npy 走 mmap）。"""
    suffix = Path(path).suffix.lower()
    if suffix == ".mat":
        try:
            import scipy.io as sio
            info = sio.whosmat(str(path))
            target = None
            for nm, shape, _ in info:
                if nm == mat_key:
                    target = shape
                    break
            if target is None:
                cands = [sh for _, sh, _ in info if len(sh) == 2]
                target = cands[0] if cands else None
            if target is not None:
                return int(max(target))          # 时间维总是较长的那一维
        except Exception:
            pass
    if suffix == ".npy":
        try:
            arr = np.load(str(path), mmap_mode="r", allow_pickle=False)
            if arr.ndim == 2:
                if arr.shape[1] == N_COLS:
                    return int(arr.shape[0])
                if arr.shape[0] == N_COLS:
                    return int(arr.shape[1])
        except Exception:
            pass
    return int(load_matrix(path, mat_key, csv_orientation, csv_header).shape[0])


# ----------------------------------------------------------------------------
# 窗口切片
# ----------------------------------------------------------------------------
def build_starts(n_points: int, input_len: int, pred_len: int, data_step: int,
                 n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """在单条轨迹上均匀撒点取窗口起点（同原实现，加了可复现的抖动）。"""
    span = (input_len + pred_len) * data_step
    usable = n_points - span
    if usable <= 0:
        return np.empty(0, dtype=np.int64)
    n = int(min(n_samples, usable))
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    block = max(usable // n, 1)
    base = np.arange(n, dtype=np.int64) * block
    jitter = rng.integers(0, block, size=n, dtype=np.int64)
    return np.clip(base + jitter, 0, usable - 1)


def make_window(feat: np.ndarray, st: int, input_len: int, pred_len: int, data_step: int):
    """
    从 (T, 10) 轨迹里切一个样本，且**只做减法、不做归一化**。

    返回 (en_delta[L,9], de_delta[H,6], tg_delta[H,6], last_abs[6])
        en_delta : 编码器输入，相对窗口末状态的位移
        de_delta : 解码器已知前缀（窗口末 H 步），同样是相对量
        tg_delta : 监督目标（窗口后 H 步），相对窗口末状态的增量
        last_abs : 窗口末状态的绝对值，推理时用于还原
    """
    L, H, S = input_len, pred_len, data_step
    last_idx = st + (L - 1) * S
    last_enc = feat[last_idx, ENC_SLICE]          # (9,) 窗口末状态（编码器视角）
    last_abs = feat[last_idx, TGT_SLICE]          # (6,) 窗口末状态（目标视角）

    en_abs = feat[st: st + L * S: S, ENC_SLICE]   # (L, 9)
    en_delta = en_abs - last_enc

    n_prefix = L - H                              # 解码器已知前缀长度
    de_abs = feat[st + n_prefix * S: st + L * S: S, TGT_SLICE]      # (H, 6)
    de_delta = de_abs - last_abs

    tg_abs = feat[st + L * S: st + (L + H) * S: S, TGT_SLICE]       # (H, 6)
    tg_delta = tg_abs - last_abs

    return (
        en_delta.astype(np.float32),
        de_delta.astype(np.float32),
        tg_delta.astype(np.float32),
        last_abs.astype(np.float32),
    )


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class TrajectoryWindowDataset(Dataset):
    """
    增量预测数据集。

    每个文件在首次访问时一次性切出全部窗口并缓存（LRU），
    因此同文件内的随机访问几乎零成本。缓存对象不可 pickle，
    已实现 __getstate__/__setstate__ 以便多进程 DataLoader 使用。
    """

    def __init__(
        self,
        files: Sequence[str],
        starts: Sequence[np.ndarray],
        normalizer: DeltaNormalizer,
        input_len: int = 40,
        pred_len: int = 20,
        data_step: int = 100,
        mat_key: str = "dataMatrix",
        csv_orientation: str = "auto",
        csv_header: str = "auto",
        cache_size: int = 8,
        tag: str = "train",
    ):
        self.files = [str(f) for f in files]
        self.starts = [np.asarray(s, dtype=np.int64) for s in starts]
        self.normalizer = normalizer
        self.input_len = input_len
        self.pred_len = pred_len
        self.data_step = data_step
        self.mat_key = mat_key
        self.csv_orientation = csv_orientation
        self.csv_header = csv_header
        self.tag = tag

        # 扁平索引: (文件序号, 窗口起点)
        self.index: List[Tuple[int, int]] = []
        self.file_span: List[Tuple[int, int]] = []   # (段起始, 段长度)，供分组采样
        for fi, arr in enumerate(self.starts):
            self.file_span.append((len(self.index), arr.shape[0]))
            for st in arr:
                self.index.append((fi, int(st)))

        self._cache_size = cache_size
        self._reset_cache()

    # -- 缓存 -------------------------------------------------------------
    def _reset_cache(self):
        self._load = lru_cache(maxsize=self._cache_size)(self._load_file)

    def __getstate__(self):
        st = self.__dict__.copy()
        st.pop("_load", None)
        return st

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reset_cache()

    def _load_file(self, fi: int) -> Dict[str, np.ndarray]:
        """切出该文件的全部窗口（原始增量空间，未归一化）。"""
        feat = load_matrix(
            self.files[fi],
            mat_key=self.mat_key,
            csv_orientation=self.csv_orientation,
            csv_header=self.csv_header,
        )
        L, H, S = self.input_len, self.pred_len, self.data_step
        n = self.starts[fi].shape[0]
        en = np.empty((n, L, ENC_DIM), np.float32)
        de = np.empty((n, H, TGT_DIM), np.float32)
        tg = np.empty((n, H, TGT_DIM), np.float32)
        last = np.empty((n, TGT_DIM), np.float32)
        for i, st in enumerate(self.starts[fi]):
            e, d, t, la = make_window(feat, int(st), L, H, S)
            en[i], de[i], tg[i], last[i] = e, d, t, la
        return {"en": en, "de": de, "tg": tg, "last": last, "feat": feat}

    # -- Dataset 接口 -----------------------------------------------------
    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int):
        fi, st = self.index[i]
        blk = self._load(fi)
        li = int(np.searchsorted(self.starts[fi], st))
        norm = self.normalizer

        en = norm.norm_en(blk["en"][li])                       # (L, 9)
        de = norm.norm_tg(blk["de"][li])                       # (H, 6)
        tg = norm.norm_tg(blk["tg"][li])                       # (H, 6)
        last = blk["last"][li]                                 # (6,)

        # 解码器输入 = [已知前缀][预测位占位 0]，长度与编码器对齐
        de_input = np.concatenate(
            [de, np.zeros((self.pred_len, TGT_DIM), np.float32)], axis=0
        )                                                       # (L, 6)

        return (
            torch.from_numpy(np.ascontiguousarray(en)).float(),
            torch.from_numpy(np.ascontiguousarray(de_input)).float(),
            torch.from_numpy(np.ascontiguousarray(tg)).float(),
            torch.from_numpy(np.ascontiguousarray(last)).float(),
        )

    # -- 辅助 -------------------------------------------------------------
    def raw_absolute(self, i: int) -> Dict[str, np.ndarray]:
        """取第 i 个样本的原始绝对量（供可视化/误差分析用）。"""
        fi, st = self.index[i]
        blk = self._load(fi)
        L, H, S = self.input_len, self.pred_len, self.data_step
        feat = blk["feat"]
        return {
            "history_abs": feat[st: st + L * S: S, TGT_SLICE].copy(),
            "future_abs": feat[st + L * S: st + (L + H) * S: S, TGT_SLICE].copy(),
            "last_abs": blk["last"][int(np.searchsorted(self.starts[fi], st))].copy(),
            "start_index": int(st),
            "file": self.files[fi],
            "timestamps": feat[st + L * S: st + (L + H) * S: S, 0].copy(),
        }


class FileGroupedSampler(Sampler):
    """
    文件分组随机采样器：每轮先打乱文件顺序，再打乱文件内样本顺序。

    目的：让同一 batch 内的样本尽量来自同一文件，从而命中 LRU 缓存，
    避免随机访问把缓存冲垮（原实现直接 shuffle=True，会反复重读 mat）。
    """

    def __init__(self, file_span: Sequence[Tuple[int, int]], seed: int = 0):
        self.file_span = [(int(a), int(n)) for a, n in file_span if n > 0]
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.file_span))
        for fi in order:
            off, n = self.file_span[fi]
            local = rng.permutation(n) + off
            for idx in local:
                yield int(idx)

    def __len__(self):
        return sum(n for _, n in self.file_span)


# ----------------------------------------------------------------------------
# 构建数据划分
# ----------------------------------------------------------------------------
def _split_lists(items, val_ratio, test_ratio, split_by, rng):
    """按文件或按目录划分 train/val/test。"""
    n = len(items)
    if n < 3:
        raise ValueError(f"数据文件太少（{n} 个），无法划分 train/val/test")
    val_n = max(1, int(round(n * val_ratio)))
    test_n = max(1, int(round(n * test_ratio)))
    if val_n + test_n >= n:
        test_n = max(1, n - val_n - 1)

    key = (lambda x: x[0]) if split_by == "dir" else (lambda x: x[1])
    groups: Dict[str, list] = {}
    for it in items:
        groups.setdefault(key(it), []).append(it)

    keys = sorted(groups.keys())
    if split_by == "dir" and len(keys) >= 3:
        perm = rng.permutation(len(keys))
        keys = [keys[i] for i in perm]
        # 先按目录尽量凑够比例
        def take(k_list, want):
            out = []
            for k in k_list:
                if len(out) >= want:
                    break
                out.extend(groups[k])
            return out

        val = take(keys, max(val_n, 1))
        rest = [k for k in keys if not set(groups[k]) & set(map(id, val))]
        test = take(rest, max(test_n, 1))
        used = set(id(x) for x in val) | set(id(x) for x in test)
        train = [it for it in items if id(it) not in used]
    else:
        perm = rng.permutation(n)
        train_idx = perm[: n - val_n - test_n]
        val_idx = perm[n - val_n - test_n: n - test_n]
        test_idx = perm[n - test_n:]
        train = [items[i] for i in train_idx]
        val = [items[i] for i in val_idx]
        test = [items[i] for i in test_idx]

    if not train or not val or not test:
        raise ValueError("数据划分失败：某个子集为空，请增加文件数或调整比例")
    return train, val, test


def build_datasets(cfg: dict, split_manifest: Optional[dict] = None, verbose: bool = True,
                   max_files: Optional[int] = None):
    """
    依配置构建 train/val/test 三个 Dataset 与归一化器。

    参数:
        cfg             : dict，至少含 data 段（见 config.json）
        split_manifest  : 已有划分清单（评估/推理时复用，保证测试集一致）
        max_files       : 每个子集最多使用几个文件（调试用，None=全部）
    返回:
        dict(datasets={train,val,test}, normalizer, manifest, dataset_map)
    """
    d = cfg["data"]
    data_root = d["data_root"]
    exts = tuple(d.get("file_types", DEFAULT_EXTS))
    mat_key = d.get("mat_key", "dataMatrix")
    csv_orientation = d.get("csv_orientation", "auto")
    csv_header = d.get("csv_header", "auto")
    input_len = int(d["input_len"])
    pred_len = int(d["pred_len"])
    data_step = int(d["data_step"])
    n_samples = int(d.get("samples_per_file", 300))
    val_ratio = float(d.get("val_ratio", 0.1))
    test_ratio = float(d.get("test_ratio", 0.1))
    split_by = d.get("split_by", "file")
    seed = int(d.get("seed", 42))
    cache_size = int(d.get("cache_size", 8))

    dataset_map = discover_datasets(data_root, d.get("datasets"), exts)
    if verbose:
        print("=" * 78)
        print("[数据] 发现数据集:")
        print(describe_datasets(dataset_map))
        print("=" * 78)

    if split_manifest is not None:
        # 复用既有划分。manifest 里存的是**相对 data_root 的路径**，
        # 换机器/换盘后只需指向新的 data_root 即可，无需改动清单本身。
        base = Path(data_root).resolve()
        split_paths = {}
        for k, v in split_manifest["splits"].items():
            split_paths[k] = [
                (str(p) if Path(p).is_absolute() else str((base / p).resolve()))
                for p in v
            ]
        if d.get("datasets"):
            allowed = {Path(p).resolve() for ps in dataset_map.values() for p in ps}
            for k in split_paths:
                split_paths[k] = [p for p in split_paths[k] if Path(p).resolve() in allowed]
        all_files = split_paths["train"] + split_paths["val"] + split_paths["test"]
        missing = [p for p in all_files if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                f"划分清单里有 {len(missing)} 个文件在 data_root 下找不到，"
                f"例如: {missing[0]}\n  请用 --data_root 指向正确的数据根目录。"
            )
        items = [(Path(p).parent.name, p) for p in all_files]
    else:
        items = []
        for name, paths in dataset_map.items():
            for p in paths:
                items.append((name, str(p)))
        items.sort(key=lambda x: x[1])

    rng = np.random.default_rng(seed)

    if split_manifest is None:
        train_items, val_items, test_items = _split_lists(
            items, val_ratio, test_ratio, split_by, rng
        )
        split_by_used = split_by
    else:
        tr_set = set(split_paths["train"])
        va_set = set(split_paths["val"])
        te_set = set(split_paths["test"])
        train_items = [it for it in items if it[1] in tr_set]
        val_items = [it for it in items if it[1] in va_set]
        test_items = [it for it in items if it[1] in te_set]
        split_by_used = split_manifest.get("split_by", split_by)

    if max_files is not None:
        k = max(1, int(max_files))
        train_items, val_items, test_items = train_items[:k], val_items[:k], test_items[:k]

    _base = Path(data_root).resolve()

    def _rel(p) -> str:
        """写成相对 data_root 的 POSIX 风格路径，便于整包搬迁。"""
        try:
            return Path(p).resolve().relative_to(_base).as_posix()
        except ValueError:
            return str(Path(p).resolve())

    manifest = {
        "data_root": str(_base),
        "datasets": list(dataset_map.keys()),
        "split_by": split_by_used,
        "seed": seed,
        "input_len": input_len,
        "pred_len": pred_len,
        "data_step": data_step,
        "samples_per_file": n_samples,
        "max_files": max_files,
        "target_space": "delta",
        "splits": {
            "train": [_rel(p) for _, p in train_items],
            "val": [_rel(p) for _, p in val_items],
            "test": [_rel(p) for _, p in test_items],
        },
    }
    if verbose:
        print(f"[数据] 划分: train={len(train_items)} / val={len(val_items)} / "
              f"test={len(test_items)} 个文件 (split_by={split_by_used})")

    def _starts_for(items_sub):
        files, starts = [], []
        for fi, (_, p) in enumerate(items_sub):
            n_points = probe_length(p, mat_key, csv_orientation, csv_header)
            st = build_starts(n_points, input_len, pred_len, data_step,
                              n_samples, np.random.default_rng(seed + fi))
            if st.size == 0:
                if verbose:
                    print(f"[跳过] 轨迹过短: {p}")
                continue
            files.append(p)
            starts.append(st)
        return files, starts

    tr_files, tr_starts = _starts_for(train_items)
    va_files, va_starts = _starts_for(val_items)
    te_files, te_starts = _starts_for(test_items)

    if not tr_files or not va_files or not te_files:
        raise ValueError("某个子集没有任何可用窗口，请减少 samples_per_file 或增加文件")

    # ---- 归一化器只在训练集上拟合 ----
    acc = DeltaNormalizer.accumulator()
    for p, sts in zip(tr_files, tr_starts):
        feat = load_matrix(p, mat_key, csv_orientation, csv_header)
        for st in sts:
            e, _, t, _ = make_window(feat, int(st), input_len, pred_len, data_step)
            acc.update(e[None], t[None])
    normalizer = acc.finalize()

    if verbose:
        print("[数据] Δ空间归一化统计（训练集）:")
        for nm, mu, sd in (("enc Δ", normalizer.mean_en, normalizer.std_en),
                           ("tgt Δ", normalizer.mean_tg, normalizer.std_tg)):
            print(f"  {nm}: mean={np.round(mu, 3).tolist()}  std={np.round(sd, 3).tolist()}")

    common = dict(input_len=input_len, pred_len=pred_len, data_step=data_step,
                  mat_key=mat_key, csv_orientation=csv_orientation,
                  csv_header=csv_header, cache_size=cache_size)
    datasets = {
        "train": TrajectoryWindowDataset(tr_files, tr_starts, normalizer, tag="train", **common),
        "val": TrajectoryWindowDataset(va_files, va_starts, normalizer, tag="val", **common),
        "test": TrajectoryWindowDataset(te_files, te_starts, normalizer, tag="test", **common),
    }
    if verbose:
        for k, ds in datasets.items():
            print(f"[数据] {k}: {len(ds):,} 个窗口 / {len(ds.files)} 个文件")

    return {"datasets": datasets, "normalizer": normalizer, "manifest": manifest,
            "dataset_map": dataset_map}


def build_loaders(cfg: dict, split_manifest: Optional[dict] = None, seed: Optional[int] = None,
                  verbose: bool = True, max_files: Optional[int] = None):
    """在 build_datasets 之上套 DataLoader。"""
    d = cfg["data"]
    built = build_datasets(cfg, split_manifest=split_manifest, verbose=verbose,
                           max_files=max_files)
    datasets = built["datasets"]
    batch_size = int(d.get("batch_size", 256))
    num_workers = int(d.get("num_workers", 0))
    seed = int(d.get("seed", 42) if seed is None else seed)
    pin = torch.cuda.is_available()
    persistent = num_workers > 0

    train_sampler = FileGroupedSampler(datasets["train"].file_span, seed=seed)
    loaders = {}
    for k, ds in datasets.items():
        if k == "train":
            loaders[k] = torch.utils.data.DataLoader(
                ds, batch_size=batch_size, sampler=train_sampler,
                num_workers=num_workers, pin_memory=pin, drop_last=True,
                persistent_workers=persistent,
            )
        else:
            loaders[k] = torch.utils.data.DataLoader(
                ds, batch_size=max(batch_size, 256), shuffle=False,
                num_workers=num_workers, pin_memory=pin,
                persistent_workers=persistent,
            )
    built["loaders"] = loaders
    built["train_sampler"] = train_sampler
    return built
