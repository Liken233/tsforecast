# -*- coding: utf-8 -*-
"""
tsforecast — 飞行器时序状态预测
================================

用编码器-解码器 Transformer 预测飞行器多变量状态序列（位置、速度、姿态角）的**增量 Δ**。

核心特性
--------
* **增量 Δ 预测**：监督量是"未来状态 − 窗口末状态"，绕开绝对量的大数值范围，
  推理时按 ``绝对 = Δ + 窗口末状态`` 还原。
* **Δ 空间全局归一化**：只在训练集上拟合 mean/std，随 checkpoint 落盘，杜绝数据泄漏。
* **编码器-解码器 Transformer**：Informer 式结构（TokenEmbedding + 正弦位置编码 +
  FullAttention）+ 三角因果掩码，逐步输出头。
* **多格式数据层**：``.mat`` / ``.csv`` / ``.npy``，多级子目录递归扫描，支持按目录/通配符挑选。
* **文件级划分 + 文件分组采样**：按文件切分训练/验证/测试防泄漏；同一 batch 尽量来自同一文件以命中缓存。

模块
----
====================  ==========================================================
``tsforecast.data``       数据层：文件发现 / 读取 / 窗口构造 / 归一化 / Dataset / Sampler
``tsforecast.models``     编码器-解码器 Transformer
``tsforecast.utils``      路径 / 种子 / 指标 / 早停 / 绘图
``tsforecast.inference``  模型装配 + 整段滚动滑窗推理
``tsforecast.train``      训练入口
``tsforecast.evaluate``   评估入口
``tsforecast.predict``    推理入口
``tsforecast.gui``        交互式可视化（Tkinter）
====================  ==========================================================

控制台命令
----------
``tsforecast-train`` / ``tsforecast-eval`` / ``tsforecast-predict`` / ``tsforecast-gui``
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
