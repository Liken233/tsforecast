import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import os
import re
import matplotlib
import threading
import time
from queue import Queue
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import torch
import torch.nn as nn
import sys
import glob
from pathlib import Path

from tsforecast.data import (
    DeltaNormalizer, load_matrix, ENC_DIM, TGT_DIM, DEFAULT_EXTS as SUPPORTED_EXTS,
)
from tsforecast.inference import load_model_bundle, predict_trajectory
from tsforecast.models import Seq2SeqTrajectoryNet as model2


# ==================== 全局配置变量 ====================
# 默认相对当前工作目录，可用命令行参数覆盖
DATA_DIR = "./data/dataset"
MODEL_DIR = "./checkpoint"
SPECIAL_DATASETS_DIR = "./checkpoint"
# ====================================================


class SingleChartViewer:
    def __init__(self, master, data_dir=None, model_dir=None, special_datasets_dir=None,
                 datasets=None, input_len=None, pred_len=None, data_step=None):
        """
        初始化单数据集图表查看器

        参数:
        - master: Tkinter主窗口
        - data_dir: 数据根目录（会递归扫描其下所有 .mat/.csv）
        - model_dir: 模型目录（其下每个子目录为一个训练产物：config.json/norm.json/checkpoint.pth）
        - special_datasets_dir: 兼容旧参数，默认为 model_dir
        - datasets: 只使用 data_dir 下的这些子目录（None=全部）
        - input_len / pred_len / data_step: 覆盖默认窗口参数
        """
        self.master = master
        master.title("Flight State Trajectory Viewer with 3D Simulation")
        master.geometry("1400x900")
        master.update_idletasks()
        try:
            master.attributes('-zoomed', True)          # X11 有效；Windows 不支持
        except tk.TclError:
            try:
                master.state('zoomed')                  # Windows 最大化
            except tk.TclError:
                pass
        # 预测显示比例
        self.prediction_keep_ratio = 1.0  # 保留前90%的预测点

        # 使用传入的目录或全局配置
        self.data_dir = data_dir if data_dir is not None else DATA_DIR
        self.model_dir = model_dir if model_dir is not None else MODEL_DIR
        self.special_datasets_dir = special_datasets_dir if special_datasets_dir is not None else SPECIAL_DATASETS_DIR
        
        # GPU配置
        self.use_gpu = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.use_gpu else 0
        self.device = torch.device(f"cuda:0" if self.use_gpu else "cpu")
        
        print(f"数据目录: {self.data_dir}")
        print(f"模型目录: {self.model_dir}")
        print(f"GPU可用: {self.use_gpu}, GPU数量: {self.gpu_count}")
        print(f"使用设备: {self.device}")

        # 数据存储
        self.datasets = {}
        self.current_dataset = None
        self.current_data_id = None

        # 模型存储
        self.models = {}
        self.current_model = None
        self.pre_time = 1
        self.pre_step = int(data_step) if data_step else 100
        self.input_len = int(input_len) if input_len else 40
        self.output_len = int(pred_len) if pred_len else 20
        self.batch_size = 256
        self.intput_dim = [0,1,2,3,4,5,6,7,8]
        self.output_dim = [3,4,5,6,7,8]
        # 只使用数据根目录下的这些子目录（None=全部）
        self.datasets_filter = datasets

        # 特殊数据集列表
        self.val_datasets = set()
        self.test_datasets = set()

        # 图表配置
        self.chart_width = 10
        self.chart_height = 6
        self.dpi = 100
        
        # 批量预测相关
        self.batch_predict_queue = Queue()
        self.batch_predict_running = False
        self.batch_results = {}

        # 逐步预测相关
        self.stepwise_predicting = False
        self.stepwise_pred_lines = []
        self.stepwise_current_index = 0
        self.stepwise_delay_ms = 50  # 调整为更合适的速度
        self.stepwise_batch_size = 10  # 调整为更小的批次
        self.stepwise_total_points = 0

        # 3D仿真相关
        self.view_mode = "2D"  # 显示模式: "2D" 或 "3D"
        self.simulation_running = False  # 3D仿真运行状态
        self.simulation_speed = 1.0  # 仿真速度
        self.current_sim_index = 0  # 当前仿真索引
        self.vehicle_model = None  # 水滴3D模型
        self.trajectory_line = None  # 轨迹线
        self.prediction_line = None  # 预测轨迹线
        self.past_trajectory = None  # 已飞行轨迹
        self.simulation_delay = 100  # 仿真延迟（毫秒）
        self.vehicle_parts = []  # 存储水滴模型的所有部件
        
        # 水滴模型参数
        self.vehicle_length = 10.0  # 水滴长度
        self.vehicle_radius = 30.0  # 水滴半径
        
        # 坐标原点偏移（用于3D显示时将起点移动到原点）
        self.origin_offset = None  # 初始化为None，将在加载数据时计算
        
        # 特征定义
        self.features_info = [
            {"name": "X Coordinate", "index": 3, "unit": "m", "color": "blue"},
            {"name": "Y Coordinate", "index": 4, "unit": "m", "color": "blue"},
            {"name": "Z Coordinate", "index": 5, "unit": "m", "color": "blue"},
            {"name": "Velocity", "index": 6, "unit": "m/s", "color": "blue"},
            {"name": "Pitch Angle", "index": 7, "unit": "°", "color": "blue"},
            {"name": "Yaw Angle", "index": 8, "unit": "°", "color": "blue"}
        ]
        
        # 显示模式
        self.show_overview = True
        self.current_feature = self.features_info[0]

        # 创建UI组件
        self.create_control_panel()
        self.create_chart_area()
        self.create_status_bar()

        # 加载数据集列表
        self.load_dataset_list()

        # 加载模型
        self.load_all_models()

    def create_control_panel(self):
        """创建控制面板"""
        control_frame = tk.Frame(self.master, padx=10, pady=10, bg="#f0f0f0")
        control_frame.pack(fill=tk.X)

        # 数据集选择框框架
        dataset_frame = tk.Frame(control_frame, bg="#f0f0f0")
        dataset_frame.pack(side=tk.LEFT, padx=20)

        tk.Label(dataset_frame, text="Select Dataset:", bg="#f0f0f0").grid(row=0, column=0, padx=5)

        self.dataset_var = tk.StringVar()
        self.dataset_selector = ttk.Combobox(dataset_frame, textvariable=self.dataset_var,
                                             width=15, state="readonly")
        self.dataset_selector.grid(row=0, column=1, padx=5)
        self.dataset_selector.bind("<<ComboboxSelected>>", self.dataset_selected)

        load_data_btn = ttk.Button(dataset_frame, text="Load Dataset", command=self.load_selected_dataset)
        load_data_btn.grid(row=0, column=2, padx=5)

        # 模型选择框框架
        model_frame = tk.Frame(control_frame, bg="#f0f0f0")
        model_frame.pack(side=tk.LEFT, padx=20)

        tk.Label(model_frame, text="Select Model:", bg="#f0f0f0").grid(row=0, column=0, padx=5)

        self.model_var = tk.StringVar()
        self.model_selector = ttk.Combobox(model_frame, textvariable=self.model_var,
                                           width=15, state="readonly")
        self.model_selector.grid(row=0, column=1, padx=5)
        self.model_selector.bind("<<ComboboxSelected>>", self.model_selected)

        time_btn = ttk.Button(model_frame, text="Set Prediction Time", command=self.set_prediction_time)
        time_btn.grid(row=0, column=2, padx=5)

        predict_btn = ttk.Button(model_frame, text="Run Prediction", command=self.predict_current_dataset)
        predict_btn.grid(row=0, column=3, padx=5)
        
        # 视图模式选择框架
        view_frame = tk.Frame(control_frame, bg="#f0f0f0")
        view_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(view_frame, text="View Mode:", bg="#f0f0f0").grid(row=0, column=0, padx=5)
        
        self.view_var = tk.StringVar(value="2D")
        self.view_selector = ttk.Combobox(view_frame, textvariable=self.view_var,
                                         width=10, state="readonly")
        self.view_selector['values'] = ("2D", "3D")
        self.view_selector.grid(row=0, column=1, padx=5)
        self.view_selector.bind("<<ComboboxSelected>>", self.view_mode_selected)
        
        # 3D仿真控制按钮
        self.sim_btn = ttk.Button(view_frame, text="Start 3D Simulation", command=self.toggle_3d_simulation)
        self.sim_btn.grid(row=0, column=2, padx=5)
        
        # 3D仿真速度设置按钮
        sim_speed_btn = ttk.Button(view_frame, text="Sim Speed", command=self.set_simulation_speed)
        sim_speed_btn.grid(row=0, column=3, padx=5)
        
        # 逐步绘制按钮
        stepwise_btn = ttk.Button(model_frame, text="Stepwise Display", command=self.toggle_stepwise_display)
        stepwise_btn.grid(row=0, column=4, padx=5)
        
        # 逐步绘制设置按钮
        stepwise_settings_btn = ttk.Button(model_frame, text="Stepwise Settings", command=self.set_stepwise_settings)
        stepwise_settings_btn.grid(row=0, column=5, padx=5)
        
        # 特征选择框架
        feature_frame = tk.Frame(control_frame, bg="#f0f0f0")
        feature_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(feature_frame, text="Select Feature:", bg="#f0f0f0").grid(row=0, column=0, padx=5)
        
        self.feature_var = tk.StringVar()
        self.feature_selector = ttk.Combobox(feature_frame, textvariable=self.feature_var,
                                             width=15, state="readonly")
        self.feature_selector.grid(row=0, column=1, padx=5)
        self.feature_selector.bind("<<ComboboxSelected>>", self.feature_selected)
        
        # 设置特征选项
        feature_names = [f["name"] for f in self.features_info]
        self.feature_selector['values'] = feature_names
        self.feature_var.set(feature_names[0])
        
        show_all_btn = ttk.Button(feature_frame, text="Show All", command=self.show_all_features)
        show_all_btn.grid(row=0, column=2, padx=5)
        
        clear_stepwise_btn = ttk.Button(feature_frame, text="Clear Stepwise", command=self.clear_stepwise_display)
        clear_stepwise_btn.grid(row=0, column=3, padx=5)

        # 导出图表按钮
        export_btn = ttk.Button(model_frame, text="Export Chart", command=self.export_chart)
        export_btn.grid(row=0, column=6, padx=5)
        
    def create_chart_area(self):
        """创建图表区域"""
        chart_frame = tk.Frame(self.master)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        frame = tk.Frame(chart_frame, bd=2, relief=tk.GROOVE)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self.title_label = tk.Label(frame, text="No dataset loaded", font=('Arial', 12, 'bold'))
        self.title_label.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # 创建matplotlib图表
        self.fig = Figure(figsize=(self.chart_width, self.chart_height), dpi=self.dpi)
        
        # 将matplotlib图表嵌入到tkinter中
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # 添加导航工具栏
        toolbar_frame = tk.Frame(frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    def create_status_bar(self):
        """创建状态栏"""
        status_frame = tk.Frame(self.master, bd=1, relief=tk.SUNKEN)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_var = tk.StringVar()
        self.status_var.set("Ready | View Mode: 2D | Prediction Time: 1 | Current Model: None | Current Dataset: None")

        status_label = tk.Label(status_frame, textvariable=self.status_var,
                                anchor=tk.W, font=('Arial', 10))
        status_label.pack(fill=tk.X, padx=10)

    def view_mode_selected(self, event=None):
        """视图模式选择事件处理"""
        old_mode = self.view_mode
        self.view_mode = self.view_var.get()
        
        # 停止3D仿真
        self.simulation_running = False
        if self.sim_btn:
            self.sim_btn.config(text="Start 3D Simulation")
        
        # 如果从3D模式切换到其他模式，进行清理
        if old_mode == "3D" and self.view_mode != "3D":
            self.clear_3d_resources()
        
        # 更新图表
        self.update_chart()
        
        # 更新状态栏
        self.status_var.set(f"View mode changed to: {self.view_mode}")

    def clear_3d_resources(self):
        """清理3D资源"""
        # 清除飞行器模型
        self.clear_vehicle_model()
        
        # 清除轨迹线
        if hasattr(self, 'past_trajectory') and self.past_trajectory is not None:
            try:
                if hasattr(self.past_trajectory, 'remove'):
                    self.past_trajectory.remove()
            except:
                pass
            finally:
                self.past_trajectory = None
        
        # 重置仿真索引
        self.current_sim_index = 0

    def toggle_3d_simulation(self):
        """切换3D仿真状态"""
        if self.view_mode != "3D":
            messagebox.showwarning("Warning", "Please switch to 3D view mode first")
            return
            
        if self.current_dataset is None:
            messagebox.showwarning("Warning", "Please load a dataset first")
            return
            
        if self.simulation_running:
            # 停止仿真
            self.simulation_running = False
            self.sim_btn.config(text="Start 3D Simulation")
            self.status_var.set("3D Simulation Stopped")
        else:
            # 开始仿真
            self.simulation_running = True
            self.current_sim_index = 0
            self.sim_btn.config(text="Stop 3D Simulation")
            self.status_var.set("3D Simulation Running")
            
            # 开始仿真循环
            self.master.after(self.simulation_delay, self.update_3d_simulation)

    def set_simulation_speed(self):
        """设置仿真速度"""
        try:
            new_speed = simpledialog.askfloat(
                "Set Simulation Speed",
                "Enter simulation speed (0.1-5.0, where 1.0 is normal speed):",
                parent=self.master,
                minvalue=0.1,
                maxvalue=5.0,
                initialvalue=self.simulation_speed
            )

            if new_speed is not None:
                self.simulation_speed = new_speed
                self.simulation_delay = int(100 / self.simulation_speed)  # 调整延迟
                self.status_var.set(f"Simulation speed set to: {self.simulation_speed:.1f}x")
        except Exception as e:
            messagebox.showerror("Setting Error", f"Invalid simulation speed: {str(e)}")

    def update_3d_simulation(self):
        """更新3D仿真"""
        if not self.simulation_running or self.current_dataset is None:
            return
            
        timestamps = self.current_dataset['timestamps']
        values = self.current_dataset['values']
        
        # 确保索引在有效范围内
        if self.current_sim_index >= len(timestamps):
            # 到达终点，重新开始或停止
            self.current_sim_index = 0
            # 可选：自动停止仿真
            # self.simulation_running = False
            # self.sim_btn.config(text="Start 3D Simulation")
            # return
            
        # 获取当前时间点的数据
        current_time = timestamps[self.current_sim_index]
        
        # 获取原始坐标
        raw_x = values[self.current_sim_index, 3]
        raw_y = values[self.current_sim_index, 4]
        raw_z = values[self.current_sim_index, 5]
        
        # 计算相对坐标（减去原点偏移）
        if self.origin_offset is not None:
            current_x = raw_x - self.origin_offset[0]
            current_y = raw_y - self.origin_offset[1]
            current_z = raw_z - self.origin_offset[2]
        else:
            current_x = raw_x
            current_y = raw_y
            current_z = raw_z
            
        current_velocity = values[self.current_sim_index, 6]
        current_pitch = values[self.current_sim_index, 7]
        current_yaw = values[self.current_sim_index, 8]
        
        # 清除旧的飞行器模型
        self.clear_vehicle_model()
        
        # 创建新的飞行器模型（使用相对坐标）
        self.create_vehicle_model(current_x, current_y, current_z, current_pitch, current_yaw)
        
        # 更新已飞行轨迹
        self.update_past_trajectory()
        
        # 更新当前索引
        self.current_sim_index += 1
        
        # 更新状态栏（显示原始坐标）
        progress_percent = (self.current_sim_index / len(timestamps)) * 100
        self.status_var.set(f"3D Simulation: {self.current_sim_index}/{len(timestamps)} points ({progress_percent:.1f}%) | "
                           f"Time: {current_time:.2f}s | Position (relative): ({current_x:.1f}, {current_y:.1f}, {current_z:.1f})m | "
                           f"Position (raw): ({raw_x:.1f}, {raw_y:.1f}, {raw_z:.1f})m | "
                           f"Speed: {current_velocity:.1f}m/s | Pitch: {current_pitch:.1f}° | Yaw: {current_yaw:.1f}°")
        
        # 重绘图表
        self.canvas.draw()
        
        # 继续仿真
        if self.simulation_running:
            self.master.after(self.simulation_delay, self.update_3d_simulation)

    def clear_vehicle_model(self):
        """清除飞行器模型"""
        # 清除所有飞行器部件
        parts_to_remove = []
        for part in self.vehicle_parts:
            if hasattr(part, 'remove'):
                try:
                    part.remove()
                except:
                    # 如果remove失败，标记为需要从列表中移除
                    parts_to_remove.append(part)
            else:
                parts_to_remove.append(part)
        
        # 从列表中移除无法删除或已删除的部件
        for part in parts_to_remove:
            if part in self.vehicle_parts:
                self.vehicle_parts.remove(part)
        
        # 确保列表是干净的
        self.vehicle_parts = [p for p in self.vehicle_parts if hasattr(p, 'remove')]

    def create_vehicle_model(self, x, y, z, pitch, yaw):
        """创建水滴3D模型（使用相对坐标）"""
        if not hasattr(self, 'ax') or self.ax is None:
            return
        
        # 清除旧模型
        self.clear_vehicle_model()
        
        # 转换角度为弧度
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
        
        # 计算飞行器方向向量
        direction = np.array([
            np.cos(pitch_rad) * np.cos(yaw_rad),
            np.cos(pitch_rad) * np.sin(yaw_rad),
            np.sin(pitch_rad)
        ])
        
        # 计算右向量和上向量
        up = np.array([0, 0, 1])
        right = np.cross(direction, up)
        if np.linalg.norm(right) < 0.001:
            right = np.array([0, 1, 0])
        else:
            right = right / np.linalg.norm(right)
        up = np.cross(right, direction)
        up = up / np.linalg.norm(up)
        
        # 创建旋转矩阵
        rotation_matrix = np.column_stack([right, up, direction])
        
        # 水滴主体 - 椭球体（更接近水滴形状）
        length = self.vehicle_length
        radius = self.vehicle_radius
        
        # 创建椭球体网格
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        
        # 椭球体参数：x方向较长（长度），y和z方向较短（半径）
        a = length / 2  # x方向半径
        b = radius      # y方向半径
        c = radius      # z方向半径
        
        # 生成椭球体表面
        x_ell = a * np.outer(np.cos(u), np.sin(v))
        y_ell = b * np.outer(np.sin(u), np.sin(v))
        z_ell = c * np.outer(np.ones_like(u), np.cos(v))
        
        # 使水滴更细长，前端更尖
        # 调整x坐标，使前端更尖
        for i in range(len(u)):
            for j in range(len(v)):
                # 根据v值调整x坐标，使前端更尖
                if z_ell[i, j] > 0:  # 前端
                    x_ell[i, j] = x_ell[i, j] * (1.0 + 0.5 * (z_ell[i, j] / c))
        
        # 变换到世界坐标系
        points = np.stack([x_ell.flatten(), y_ell.flatten(), z_ell.flatten()], axis=1)
        points_rotated = np.dot(points, rotation_matrix.T)
        
        # 应用平移（相对坐标）
        points_rotated[:, 0] += x
        points_rotated[:, 1] += y
        points_rotated[:, 2] += z
        
        # 重塑为网格
        x_final = points_rotated[:, 0].reshape(x_ell.shape)
        y_final = points_rotated[:, 1].reshape(y_ell.shape)
        z_final = points_rotated[:, 2].reshape(z_ell.shape)
        
        # 绘制水滴主体
        body = self.ax.plot_surface(x_final, y_final, z_final, 
                                   color='cyan', alpha=0.8, 
                                   edgecolor='darkblue', linewidth=0.5,
                                   shade=True)
        self.vehicle_parts.append(body)
        
        # 水滴头部 - 更尖的部分
        head_radius = radius * 0.8
        head_length = length * 0.2
        
        # 创建头部网格
        u_head = np.linspace(0, 2 * np.pi, 20)
        v_head = np.linspace(0, np.pi/2, 10)
        
        x_head = head_length * np.outer(np.cos(u_head), np.sin(v_head))
        y_head = head_radius * np.outer(np.sin(u_head), np.sin(v_head))
        z_head = head_radius * np.outer(np.ones_like(u_head), np.cos(v_head))
        
        # 头部位于水滴前端
        x_head = x_head + length/2
        
        # 变换到世界坐标系
        head_points = np.stack([x_head.flatten(), y_head.flatten(), z_head.flatten()], axis=1)
        head_points_rotated = np.dot(head_points, rotation_matrix.T)
        head_points_rotated[:, 0] += x
        head_points_rotated[:, 1] += y
        head_points_rotated[:, 2] += z
        
        # 重塑为网格
        x_head_final = head_points_rotated[:, 0].reshape(x_head.shape)
        y_head_final = head_points_rotated[:, 1].reshape(y_head.shape)
        z_head_final = head_points_rotated[:, 2].reshape(z_head.shape)
        
        # 绘制水滴头部
        head = self.ax.plot_surface(x_head_final, y_head_final, z_head_final, 
                                  color='lightblue', alpha=0.9,
                                  edgecolor='blue', linewidth=0.3,
                                  shade=True)
        self.vehicle_parts.append(head)
        
        # 添加尾翼（简化版，小三角形）
        wing_size = radius * 1.5
        
        # 创建4个小尾翼
        wing_positions = [
            (0, wing_size, -length/3),    # 右翼
            (0, -wing_size, -length/3),   # 左翼
            (wing_size, 0, -length/3),    # 上翼
            (-wing_size, 0, -length/3)    # 下翼
        ]
        
        wing_colors = ['red', 'green', 'yellow', 'orange']
        
        for i, (wx, wy, wz) in enumerate(wing_positions):
            # 创建三角形翼面
            wing_x = np.array([[wx, wx + wing_size/2, wx - wing_size/2]])
            wing_y = np.array([[wy, wy, wy]])
            wing_z = np.array([[wz, wz + wing_size/2, wz + wing_size/2]])
            
            # 变换到世界坐标系
            wing_points = np.stack([wing_x.flatten(), wing_y.flatten(), wing_z.flatten()], axis=1)
            wing_points_rotated = np.dot(wing_points, rotation_matrix.T)
            wing_points_rotated[:, 0] += x
            wing_points_rotated[:, 1] += y
            wing_points_rotated[:, 2] += z
            
            # 重塑为网格
            wing_x_final = wing_points_rotated[:, 0].reshape(wing_x.shape)
            wing_y_final = wing_points_rotated[:, 1].reshape(wing_y.shape)
            wing_z_final = wing_points_rotated[:, 2].reshape(wing_z.shape)
            
            # 绘制翼面
            wing = self.ax.plot_surface(wing_x_final, wing_y_final, wing_z_final, 
                                      color=wing_colors[i % len(wing_colors)], alpha=0.7,
                                      edgecolor='black', linewidth=0.2,
                                      shade=True)
            self.vehicle_parts.append(wing)

    def update_past_trajectory(self):
        """更新已飞行轨迹（使用相对坐标）"""
        if not hasattr(self, 'ax') or self.ax is None:
            return
        
        if self.current_dataset is None or self.origin_offset is None:
            return
        
        values = self.current_dataset['values']
        
        # 获取已飞行的轨迹点
        if self.current_sim_index > 0:
            # 获取原始坐标
            raw_x = values[:self.current_sim_index, 3]
            raw_y = values[:self.current_sim_index, 4]
            raw_z = values[:self.current_sim_index, 5]
            
            # 转换为相对坐标
            past_x = raw_x - self.origin_offset[0]
            past_y = raw_y - self.origin_offset[1]
            past_z = raw_z - self.origin_offset[2]
            
            # 清除旧的已飞行轨迹
            if hasattr(self, 'past_trajectory') and self.past_trajectory is not None:
                try:
                    # 使用正确的方法移除线条
                    if hasattr(self.past_trajectory, 'remove'):
                        self.past_trajectory.remove()
                    elif hasattr(self.ax, 'lines') and self.past_trajectory in self.ax.lines:
                        # 从列表中移除
                        index = None
                        for i, line in enumerate(self.ax.lines):
                            if line is self.past_trajectory:
                                index = i
                                break
                        if index is not None:
                            del self.ax.lines[index]
                except Exception as e:
                    print(f"移除轨迹线时出错: {e}")
                finally:
                    self.past_trajectory = None
            
            # 绘制新的已飞行轨迹
            if len(past_x) > 1:
                self.past_trajectory = self.ax.plot(past_x, past_y, past_z, 
                                                'g-', linewidth=2, alpha=0.8, 
                                                label='Flown Path')[0]
            
    def clear_stepwise_display(self):
        """清除逐步绘制结果"""
        self.stepwise_predicting = False
        self.stepwise_pred_lines = []
        self.stepwise_current_index = 0
        self.stepwise_total_points = 0
        
        if self.current_dataset and self.current_dataset['predictions'] is not None:
            self.status_var.set("逐步绘制已清除 | 预测数据仍保留")
        else:
            self.status_var.set("逐步绘制已清除")
        
        self.update_chart()
        messagebox.showinfo("Clear Stepwise", "逐步绘制结果已清除")

    def set_stepwise_settings(self):
        """设置逐步绘制参数"""
        settings_window = tk.Toplevel(self.master)
        settings_window.title("Stepwise Display Settings")
        settings_window.geometry("400x250")
        settings_window.transient(self.master)
        settings_window.grab_set()
        
        settings_window.update_idletasks()
        x = (settings_window.winfo_screenwidth() - settings_window.winfo_reqwidth()) // 2
        y = (settings_window.winfo_screenheight() - settings_window.winfo_reqheight()) // 2
        settings_window.geometry(f"+{x}+{y}")
        
        title_label = tk.Label(settings_window, text="Stepwise Display Settings", font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)
        
        batch_size_frame = tk.Frame(settings_window)
        batch_size_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(batch_size_frame, text="Points per batch:", font=('Arial', 10)).pack(side=tk.LEFT)
        
        self.batch_size_var = tk.StringVar(value=str(self.stepwise_batch_size))
        batch_size_entry = tk.Entry(batch_size_frame, textvariable=self.batch_size_var, width=10)
        batch_size_entry.pack(side=tk.LEFT, padx=10)
        
        tk.Label(batch_size_frame, text="(1-1000)", font=('Arial', 9, 'italic')).pack(side=tk.LEFT)
        
        delay_frame = tk.Frame(settings_window)
        delay_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(delay_frame, text="Delay between batches:", font=('Arial', 10)).pack(side=tk.LEFT)
        
        self.delay_var = tk.StringVar(value=str(self.stepwise_delay_ms))
        delay_entry = tk.Entry(delay_frame, textvariable=self.delay_var, width=10)
        delay_entry.pack(side=tk.LEFT, padx=10)
        
        tk.Label(delay_frame, text="ms (1-1000)", font=('Arial', 9, 'italic')).pack(side=tk.LEFT)
        
        info_label = tk.Label(settings_window, 
                              text="提示: 点数越多绘制越快，延迟越小动画越快",
                              font=('Arial', 9, 'italic'), fg="blue")
        info_label.pack(pady=5)
        
        preview_frame = tk.Frame(settings_window)
        preview_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.preview_var = tk.StringVar(value="")
        preview_label = tk.Label(preview_frame, textvariable=self.preview_var, font=('Arial', 9))
        preview_label.pack()
        
        def update_preview():
            try:
                batch_size = int(self.batch_size_var.get())
                delay = int(self.delay_var.get())
                
                if self.current_dataset and self.current_dataset['predictions'] is not None:
                    total_points = len(self.current_dataset['predictions']['timestamps'])
                    if total_points > 0:
                        batches = (total_points + batch_size - 1) // batch_size
                        total_time = (batches * delay) / 1000
                        self.preview_var.set(f"预计: {batches}批, 总时间: {total_time:.1f}秒")
                    else:
                        self.preview_var.set("当前无预测数据")
                else:
                    self.preview_var.set("请先运行预测")
            except:
                self.preview_var.set("请输入有效数字")
        
        self.batch_size_var.trace("w", lambda *args: update_preview())
        self.delay_var.trace("w", lambda *args: update_preview())
        
        update_preview()
        
        button_frame = tk.Frame(settings_window)
        button_frame.pack(pady=20)
        
        def apply_settings():
            try:
                batch_size = int(self.batch_size_var.get())
                delay = int(self.delay_var.get())
                
                if batch_size < 1 or batch_size > 1000:
                    messagebox.showerror("错误", "每批点数必须在1-1000之间")
                    return
                
                if delay < 1 or delay > 1000:
                    messagebox.showerror("错误", "延迟时间必须在1-1000毫秒之间")
                    return
                
                self.stepwise_batch_size = batch_size
                self.stepwise_delay_ms = delay
                
                if self.stepwise_predicting:
                    messagebox.showinfo("设置已更新", 
                                      f"逐步绘制设置已更新:\n每批点数: {batch_size}\n延迟: {delay}ms")
                else:
                    messagebox.showinfo("设置已保存", 
                                      f"逐步绘制设置已保存:\n每批点数: {batch_size}\n延迟: {delay}ms")
                
                settings_window.destroy()
                
            except ValueError:
                messagebox.showerror("错误", "请输入有效的数字")
        
        def cancel_settings():
            settings_window.destroy()
        
        apply_btn = tk.Button(button_frame, text="应用", command=apply_settings, width=10)
        apply_btn.pack(side=tk.LEFT, padx=10)
        
        cancel_btn = tk.Button(button_frame, text="取消", command=cancel_settings, width=10)
        cancel_btn.pack(side=tk.LEFT, padx=10)

    def toggle_stepwise_display(self):
        """切换逐步绘制模式"""
        if self.stepwise_predicting:
            self.stepwise_predicting = False
            self.status_var.set(f"逐步绘制已停止 | 绘制了 {self.stepwise_current_index}/{self.stepwise_total_points} 点")
            messagebox.showinfo("Stepwise Display", "逐步绘制已停止")
        else:
            if self.current_dataset and self.current_dataset['predictions'] is not None:
                self.stepwise_predicting = True
                self.start_stepwise_display()
            else:
                messagebox.showwarning("Warning", "请先运行预测以获取预测结果")

    def start_stepwise_display(self):
        """开始逐步绘制预测结果"""
        if not self.current_dataset or self.current_dataset['predictions'] is None:
            return
        
        self.stepwise_total_points = len(self.current_dataset['predictions']['timestamps'])
        self.stepwise_current_index = 0
        self.stepwise_pred_lines = []
        
        if self.show_overview:
            for ax in self.fig.axes:
                lines_to_remove = []
                for line in ax.lines:
                    if 'Pred' in line.get_label():
                        lines_to_remove.append(line)
                for line in lines_to_remove:
                    line.remove()
        else:
            if hasattr(self, 'ax'):
                lines_to_remove = []
                for line in self.ax.lines:
                    if 'Pred' in line.get_label():
                        lines_to_remove.append(line)
                for line in lines_to_remove:
                    line.remove()
        
        self.update_chart_title_with_progress()
        self.canvas.draw()
        self.stepwise_draw_next_batch()

    def stepwise_draw_next_batch(self):
        """逐步绘制下一批预测点"""
        if not self.stepwise_predicting:
            return
        
        pred_timestamps = self.current_dataset['predictions']['timestamps']
        pred_values = self.current_dataset['predictions']['values']
        
        end_index = min(self.stepwise_current_index + self.stepwise_batch_size, len(pred_timestamps))
        
        if self.stepwise_current_index >= len(pred_timestamps):
            self.stepwise_predicting = False
            self.status_var.set(f"逐步绘制完成 | 共绘制了 {len(pred_timestamps)} 个预测点")
            self.update_chart_title_with_progress(completed=True)
            messagebox.showinfo("Stepwise Display", f"逐步绘制完成！\n共绘制了 {len(pred_timestamps)} 个预测点")
            return
        
        if self.view_mode == "2D":
            if self.show_overview:
                axes = self.fig.axes
                for i, ax in enumerate(axes):
                    if i < pred_values.shape[1]:
                        current_timestamps = pred_timestamps[self.stepwise_current_index:end_index]
                        current_values = pred_values[self.stepwise_current_index:end_index, i]
                        
                        if len(current_timestamps) > 0:
                            line = ax.plot(current_timestamps, current_values,
                                          'ro', markersize=1.5, alpha=0.6, 
                                          label=f'Pred (t+{self.pre_time})' if self.stepwise_current_index == 0 else "")
                            self.stepwise_pred_lines.append(line[0])
                            
                            if end_index >= len(pred_timestamps):
                                ax.legend(loc='best', fontsize=8)
            else:
                if hasattr(self, 'ax'):
                    pred_index = self.features_info.index(self.current_feature)
                    if pred_index < pred_values.shape[1]:
                        current_timestamps = pred_timestamps[self.stepwise_current_index:end_index]
                        current_values = pred_values[self.stepwise_current_index:end_index, pred_index]
                        
                        if len(current_timestamps) > 0:
                            line = self.ax.plot(current_timestamps, current_values,
                                              'ro', markersize=5, alpha=0.7, 
                                              label=f'Prediction (t+{self.pre_time})' if self.stepwise_current_index == 0 else "")
                            self.stepwise_pred_lines.append(line[0])
                            
                            if end_index >= len(pred_timestamps):
                                self.ax.legend(loc='best', fontsize=10)
        elif self.view_mode == "3D":
            if hasattr(self, 'ax'):
                # 绘制预测轨迹（使用相对坐标）
                if self.origin_offset is not None:
                    # 预测数据已经是在原始坐标系中，需要转换为相对坐标
                    raw_x = pred_values[self.stepwise_current_index:end_index, 0]
                    raw_y = pred_values[self.stepwise_current_index:end_index, 1]
                    raw_z = pred_values[self.stepwise_current_index:end_index, 2]
                    
                    current_x = raw_x - self.origin_offset[0]
                    current_y = raw_y - self.origin_offset[1]
                    current_z = raw_z - self.origin_offset[2]
                else:
                    current_x = pred_values[self.stepwise_current_index:end_index, 0]
                    current_y = pred_values[self.stepwise_current_index:end_index, 1]
                    current_z = pred_values[self.stepwise_current_index:end_index, 2]
                
                if len(current_x) > 0:
                    line = self.ax.plot(current_x, current_y, current_z,
                                      'r-', linewidth=2, alpha=0.7,
                                      label='Prediction' if self.stepwise_current_index == 0 else "")
                    self.stepwise_pred_lines.append(line[0])
                    
                    if end_index >= len(pred_timestamps):
                        self.ax.legend(loc='best')
        
        self.stepwise_current_index = end_index
        progress_percent = (self.stepwise_current_index / len(pred_timestamps)) * 100
        self.status_var.set(f"逐步绘制: {self.stepwise_current_index}/{len(pred_timestamps)} 点 ({progress_percent:.1f}%) | "
                           f"批次大小: {self.stepwise_batch_size}点 | 延迟: {self.stepwise_delay_ms}ms")
        
        self.update_chart_title_with_progress()
        self.canvas.draw()
        
        if self.stepwise_current_index < len(pred_timestamps):
            self.master.after(self.stepwise_delay_ms, self.stepwise_draw_next_batch)
        else:
            self.stepwise_predicting = False
            self.status_var.set(f"逐步绘制完成 | 共绘制了 {len(pred_timestamps)} 个预测点")
            self.update_chart_title_with_progress(completed=True)
            messagebox.showinfo("Stepwise Display", f"逐步绘制完成！\n共绘制了 {len(pred_timestamps)} 个预测点")

    def update_chart_title_with_progress(self, completed=False):
        """更新图表标题显示进度"""
        if not self.current_dataset:
            return
        
        dataset_status = ""
        if self.current_data_id in self.test_datasets:
            dataset_status = " (TEST)"
            title_color = 'red'
        elif self.current_data_id in self.val_datasets:
            dataset_status = " (VAL)"
            title_color = 'orange'
        else:
            title_color = 'black'
        
        base_title = f"Dataset {self.current_data_id}{dataset_status} - {self.current_dataset['file']}"
        
        if completed:
            progress_text = " [逐步绘制完成]"
        elif self.stepwise_predicting:
            progress_percent = (self.stepwise_current_index / self.stepwise_total_points) * 100
            progress_text = f" [逐步绘制中: {self.stepwise_current_index}/{self.stepwise_total_points} ({progress_percent:.1f}%)]"
        else:
            progress_text = ""
        
        view_text = f" [{self.view_mode} View]"
        
        if self.show_overview:
            self.title_label.config(text=f"{base_title} [Overview Mode]{view_text}{progress_text}", 
                                  fg=title_color, font=('Arial', 12, 'bold'))
        else:
            selected_feature = self.current_feature
            self.title_label.config(text=f"{base_title} [{selected_feature['name']}]{view_text}{progress_text}", 
                                  fg=title_color, font=('Arial', 12, 'bold'))

    def load_dataset_list(self):
        """递归扫描数据根目录，列出所有 .mat / .csv 轨迹"""
        if not os.path.exists(self.data_dir):
            self.status_var.set(f"Error: Data directory not found - {self.data_dir}")
            messagebox.showerror("Directory Error", f"Data directory not found:\n{self.data_dir}")
            return

        try:
            files = []
            for root, dirs, names in os.walk(self.data_dir):
                rel_dir = os.path.relpath(root, self.data_dir).replace('\\', '/')
                if self.datasets_filter and rel_dir != '.':
                    top = rel_dir.split('/')[0]
                    if top not in self.datasets_filter and rel_dir not in self.datasets_filter:
                        dirs[:] = []
                        continue
                for nm in names:
                    if nm.lower().endswith(SUPPORTED_EXTS) and not nm.startswith('.'):
                        files.append(os.path.join(root, nm))
            files.sort()

        except Exception as e:
            self.status_var.set(f"File search error: {str(e)}")
            messagebox.showerror("File Error", f"Error finding data files:\n{str(e)}")
            return

        dataset_ids = []
        self.id_to_file = {}

        for fp in files:
            rel = os.path.relpath(fp, self.data_dir).replace('\\', '/')
            file_id = os.path.splitext(rel)[0]
            dataset_ids.append(file_id)
            self.id_to_file[file_id] = fp

        self.load_special_datasets()

        def _natural(s):
            return [int(t) if t.isdigit() else t.lower()
                    for t in re.split(r'(\d+)', s)]

        dataset_list = sorted(dataset_ids, key=_natural)
        self.dataset_selector['values'] = dataset_list

        if dataset_list:
            self.dataset_var.set(dataset_list[0])

        self.status_var.set(
            f"Found {len(dataset_list)} trajectories under {self.data_dir} | "
            f"View Mode: {self.view_mode} | Prediction Time: {self.pre_time}")

    def load_special_datasets(self):
        """从各模型目录的 split.json / val.txt / test.txt 汇总验证集与测试集清单"""
        self.val_datasets = set()
        self.test_datasets = set()
        if not os.path.isdir(self.model_dir):
            return

        def _key(path_like):
            p = str(path_like)
            try:
                rel = os.path.relpath(p, self.data_dir)
            except ValueError:
                rel = p
            rel = rel.replace('\\', '/')
            if rel.startswith('../'):
                rel = os.path.basename(p)
            return os.path.splitext(rel)[0]

        for name in sorted(os.listdir(self.model_dir)):
            d = os.path.join(self.model_dir, name)
            if not os.path.isdir(d):
                continue

            split_json = os.path.join(d, 'split.json')
            if os.path.exists(split_json):
                try:
                    with open(split_json, 'r', encoding='utf-8') as f:
                        splits = json.load(f).get('splits', {})
                    for kind, bucket in (('val', self.val_datasets), ('test', self.test_datasets)):
                        for p in splits.get(kind, []):
                            bucket.add(_key(p))
                except Exception as e:
                    print(f"读取 split.json 失败 {split_json}: {e}")

            for fn, bucket in (('val.txt', self.val_datasets), ('test.txt', self.test_datasets)):
                fp = os.path.join(d, fn)
                if not os.path.exists(fp):
                    continue
                try:
                    with open(fp, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    if content:
                        bucket.update(v.strip() for v in content.split(',') if v.strip())
                except Exception as e:
                    print(f"读取 {fn} 失败: {e}")

    def load_selected_dataset(self):
        """加载选中的轨迹（支持 .mat / .csv）"""
        selected = self.dataset_var.get()
        if not selected:
            messagebox.showwarning("Warning", "Please select a dataset first")
            return

        if selected not in self.id_to_file:
            messagebox.showerror("Error", f"Dataset {selected} not found")
            return

        file_path = self.id_to_file[selected]

        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"File not found: {file_path}")
            return

        try:
            feat = load_matrix(file_path)
            timestamps = feat[:, 0]
            values = feat[:, 1:]

            # 计算原点偏移（第一个数据点的坐标）
            start_x = values[0, 3]
            start_y = values[0, 4]
            start_z = values[0, 5]
            self.origin_offset = np.array([start_x, start_y, start_z])

            print(f"原始起始点坐标: ({start_x:.2f}, {start_y:.2f}, {start_z:.2f})")
            print(f"原点偏移: {self.origin_offset}")

            self.current_data_id = selected
            self.current_dataset = {
                'file': os.path.basename(file_path),
                'timestamps': timestamps,
                'values': values,
                'min': np.min(values),
                'max': np.max(values),
                'predictions': None,
                'start_point': (start_x, start_y, start_z)  # 保存原始起始点
            }

            self.stepwise_predicting = False
            self.stepwise_pred_lines = []
            self.stepwise_current_index = 0
            self.stepwise_total_points = 0

            self.simulation_running = False
            self.current_sim_index = 0

            self.update_chart()

            self.status_var.set(
                f"Loaded dataset {selected} | View Mode: {self.view_mode} | Prediction Time: {self.pre_time} | "
                f"Model: {self.model_var.get() if self.model_var.get() else 'None'} | "
                f"Feature: {self.feature_var.get()} | "
                f"Origin offset: ({self.origin_offset[0]:.1f}, {self.origin_offset[1]:.1f}, {self.origin_offset[2]:.1f})")

        except Exception as e:
            messagebox.showerror("Load Error", f"Error loading dataset: {str(e)}")
            import traceback
            traceback.print_exc()

    def dataset_selected(self, event=None):
        """数据集选择事件处理"""
        pass

    def feature_selected(self, event=None):
        """特征选择事件处理"""
        self.show_overview = False
        self.stepwise_predicting = False
        self.stepwise_pred_lines = []
        self.stepwise_current_index = 0
        self.stepwise_total_points = 0
        
        selected_name = self.feature_var.get()
        for feature in self.features_info:
            if feature["name"] == selected_name:
                self.current_feature = feature
                break
        
        self.update_chart()
        
        if self.current_data_id:
            self.status_var.set(
                f"Dataset {self.current_data_id} | View Mode: {self.view_mode} | Prediction Time: {self.pre_time} | "
                f"Model: {self.current_model if self.current_model else 'None'} | "
                f"Feature: {selected_name}")

    def show_all_features(self):
        """显示所有特征（概览模式）"""
        self.show_overview = True
        self.stepwise_predicting = False
        self.stepwise_pred_lines = []
        self.stepwise_current_index = 0
        self.stepwise_total_points = 0
        
        self.update_chart()
        
        if self.current_data_id:
            self.status_var.set(
                f"Dataset {self.current_data_id} | View Mode: {self.view_mode} | Prediction Time: {self.pre_time} | "
                f"Model: {self.current_model if self.current_model else 'None'} | "
                f"Feature: All Features (Overview)")

    def load_all_models(self):
        """加载所有模型 —— 只认新格式产物目录（config.json + norm.json + *.pth）"""
        if not os.path.exists(self.model_dir):
            self.status_var.set(f"Error: Model directory not found - {self.model_dir}")
            messagebox.showerror("Directory Error", f"Model directory not found:\n{self.model_dir}")
            return

        self.models = {}

        model_dirs = []
        for item in sorted(os.listdir(self.model_dir)):
            item_path = os.path.join(self.model_dir, item)
            if not os.path.isdir(item_path):
                continue
            cfg_path = os.path.join(item_path, 'config.json')
            norm_path = os.path.join(item_path, 'norm.json')
            ckpt_path = os.path.join(item_path, 'checkpoint.pth')
            if not os.path.exists(ckpt_path):
                pth = sorted(glob.glob(os.path.join(item_path, '*.pth')))
                ckpt_path = pth[0] if pth else None
            if ckpt_path and os.path.exists(cfg_path) and os.path.exists(norm_path):
                model_dirs.append((item, ckpt_path, norm_path, cfg_path))
            else:
                print(f"[跳过] {item}: 需要 config.json + norm.json + *.pth 三者齐全"
                      f"（由 train.py 产出的新格式目录）")

        for i, (model_name, ckpt_path, norm_path, cfg_path) in enumerate(model_dirs):
            model_id = f"model_{i + 1}"
            try:
                model, normalizer, cfg = load_model_bundle(
                    os.path.dirname(ckpt_path),
                    ckpt_name=os.path.basename(ckpt_path),
                    device=self.device,
                )
                params = {
                    'input_dim': ENC_DIM,
                    'd_model': cfg['model']['d_model'],
                    'num_heads': cfg['model']['num_heads'],
                    'pred_dim': TGT_DIM,
                    'pre_len': cfg['data']['pred_len'],
                    'layer_num': cfg['model']['layer_num'],
                    'dropout': cfg['model']['dropout'],
                    'input_len': cfg['data']['input_len'],
                    'data_step': cfg['data']['data_step'],
                }
                self.models[model_id] = {
                    'name': model_name,
                    'model': model,
                    'path': ckpt_path,
                    'norm_path': norm_path,
                    'config_path': cfg_path,
                    'normalizer': normalizer,
                    'params': params,
                    'config': cfg,
                }
                print(f"成功加载模型: {model_name}  (d_model={params['d_model']}, "
                      f"layers={params['layer_num']}, L={params['input_len']}, "
                      f"H={params['pre_len']}, step={params['data_step']})")

            except Exception as e:
                print(f"模型加载错误: {model_name} - {str(e)}")
                import traceback
                traceback.print_exc()

        model_names = [info['name'] for info in self.models.values()]
        self.model_selector['values'] = model_names

        if model_names:
            self.model_var.set(model_names[0])
            self.model_selected()

        self.status_var.set(f"Loaded {len(model_names)} models | View Mode: {self.view_mode} | Prediction Time: {self.pre_time}")

    def model_selected(self, event=None):
        """模型选择事件处理：同时采用该模型训练时的窗口参数 L / H / data_step"""
        selected = self.model_var.get()
        if selected:
            self.current_model = selected
            for info in self.models.values():
                if info['name'] == selected:
                    p = info['params']
                    self.input_len = int(p.get('input_len', self.input_len))
                    self.output_len = int(p.get('pre_len', self.output_len))
                    self.pre_step = int(p.get('data_step', self.pre_step))
                    if self.pre_time > self.output_len:
                        self.pre_time = self.output_len
                    break
            self.status_var.set(
                f"Selected model: {selected} (L={self.input_len}, H={self.output_len}, "
                f"step={self.pre_step}) | View Mode: {self.view_mode} | Prediction Time: {self.pre_time} | "
                f"Dataset: {self.current_data_id if self.current_data_id else 'None'} | "
                f"Feature: {self.feature_var.get()}")

    def set_prediction_time(self):
        """设置预测时间步长"""
        try:
            new_time = simpledialog.askinteger(
                "Set Prediction Time",
                f"Enter prediction time (1-{self.output_len}):",
                parent=self.master,
                minvalue=1,
                maxvalue=self.output_len,
                initialvalue=self.pre_time
            )

            if new_time is not None:
                self.pre_time = new_time
                self.status_var.set(
                    f"Prediction time set to: {self.pre_time} | View Mode: {self.view_mode} | "
                    f"Model: {self.current_model if self.current_model else 'None'} | "
                    f"Dataset: {self.current_data_id if self.current_data_id else 'None'} | "
                    f"Feature: {self.feature_var.get()}")

                if self.current_dataset and self.current_dataset['predictions'] is not None:
                    self.update_chart()
        except Exception as e:
            messagebox.showerror("Setting Error", f"Invalid prediction time: {str(e)}")

    def predict_single_dataset(self, dataset_id, model_info, progress_callback=None):
        """对单条轨迹做滚动滑窗预测（增量模型 + Δ空间全局归一化），返回绝对轨迹"""
        if model_info is None:
            return None

        file_path = self.id_to_file.get(dataset_id)
        if file_path is None or not os.path.exists(file_path):
            print(f"数据集 {dataset_id} 不存在: {file_path}")
            return None

        try:
            feat = load_matrix(file_path)                      # (T, 10)

            params = model_info['params']
            L = int(params.get('input_len', self.input_len))
            H = int(params.get('pre_len', self.output_len))
            S = int(params.get('data_step', self.pre_step))

            if len(feat) < (L + H) * S + 1:
                print(f"数据集 {dataset_id} 数据点不足（{len(feat)} < {(L + H) * S + 1}）")
                return None

            horizon = min(int(self.pre_time), H)

            res = predict_trajectory(
                model_info['model'], model_info['normalizer'], feat,
                input_len=L, pred_len=H, data_step=S,
                device=self.device, horizon=horizon,
                batch_size=self.batch_size,
                keep_ratio=self.prediction_keep_ratio,
            )

            pred_values = res['values']               # 绝对轨迹 [m, 6]
            pred_timestamps = res['timestamps']
            points_idx = res['points_idx']

            print(f"数据集 {dataset_id}: 共 {res['full_pred'].shape[0]} 个窗口，"
                  f"展示 {len(pred_values)} 个点，horizon={horizon}")

            return {
                'dataset_id': dataset_id,
                'timestamps': pred_timestamps,
                'points_idx': points_idx,
                'values': pred_values,
                'truths': res['truths'],
                'file': os.path.basename(file_path),
            }

        except Exception as e:
            print(f"数据集 {dataset_id} 预测错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_current_dataset(self):
        """对当前数据集运行预测"""
        if self.current_model is None:
            messagebox.showwarning("Warning", "Please select a model first")
            return

        if self.current_dataset is None:
            messagebox.showwarning("Warning", "Please load a dataset first")
            return

        model_info = None
        for info in self.models.values():
            if info['name'] == self.current_model:
                model_info = info
                break

        if model_info is None:
            messagebox.showerror("Error", "Selected model is not available")
            return

        progress_window = tk.Toplevel(self.master)
        progress_window.title("Running Prediction")
        progress_window.geometry("300x150")
        progress_window.transient(self.master)
        progress_window.grab_set()

        progress_label = tk.Label(progress_window, text=f"Running prediction for dataset {self.current_data_id}...")
        progress_label.pack(pady=10)

        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, padx=20, pady=5)
        
        status_label = tk.Label(progress_window, text="Initializing...")
        status_label.pack(pady=5)

        def update_progress(current, total):
            percent = (current / total) * 100
            progress_var.set(percent)
            status_label.config(text=f"Processing batch {current}/{total}")
            progress_window.update()

        def run_prediction():
            try:
                result = self.predict_single_dataset(
                    self.current_data_id, 
                    model_info, 
                    progress_callback=update_progress
                )
                
                if result is not None:
                    #############################################################
                    # 更新状态信息，显示去除了多少点
                    total_pred_points = len(result['timestamps'])
                    original_points = total_pred_points / self.prediction_keep_ratio if total_pred_points > 0 else 0
                    removed_points = original_points - total_pred_points
                    
                    self.current_dataset['predictions'] = {
                        'timestamps': result['timestamps'],
                        'values': result['values']
                    }
                    
                    self.stepwise_predicting = False
                    self.stepwise_pred_lines = []
                    self.stepwise_current_index = 0
                    self.stepwise_total_points = 0
                    
                    self.master.after(0, lambda: self.update_chart())
                    self.master.after(0, lambda: self.status_var.set(
                        f"预测完成 | 显示 {total_pred_points} 个点 (去除最后 {removed_points:.0f} 个点) | "
                        f"视图模式: {self.view_mode} | 预测时间: {self.pre_time} | "
                        f"模型: {self.current_model} | 数据集: {self.current_data_id} | "
                        f"特征: {self.feature_var.get()}"))
                    
                    self.master.after(100, lambda: self.ask_for_stepwise_display())
                else:
                    self.master.after(0, lambda: messagebox.showerror("Prediction Error", f"Error during prediction"))
                
            except Exception as e:
                self.master.after(0, lambda: messagebox.showerror("Prediction Error", f"Error during prediction: {str(e)}"))
            finally:
                self.master.after(0, progress_window.destroy)

        prediction_thread = threading.Thread(target=run_prediction)
        prediction_thread.daemon = True
        prediction_thread.start()

    def ask_for_stepwise_display(self):
        """询问是否要开始逐步绘制"""
        total_points = len(self.current_dataset['predictions']['timestamps'])
        if total_points > 0:
            batches = (total_points + self.stepwise_batch_size - 1) // self.stepwise_batch_size
            total_time = (batches * self.stepwise_delay_ms) / 1000
            
            response = messagebox.askyesno("Stepwise Display", 
                                          f"预测已完成！共{total_points}个预测点。\n\n"
                                          f"当前设置:\n"
                                          f"- 每批点数: {self.stepwise_batch_size}\n"
                                          f"- 延迟: {self.stepwise_delay_ms}ms\n"
                                          f"- 预计批数: {batches}\n"
                                          f"- 预计时间: {total_time:.1f}秒\n\n"
                                          f"是否要以逐步方式显示预测结果？\n"
                                          f"(您可以在开始前点击'Stepwise Settings'调整设置)")
            if response:
                self.stepwise_predicting = True
                self.start_stepwise_display()
        else:
            messagebox.showinfo("Prediction Complete", "预测已完成，但没有预测点可以显示。")

    def update_chart(self):
        """更新图表 - 支持2D曲线和3D仿真模式"""
        self.fig.clf()
        
        if self.current_dataset is None:
            self.ax = self.fig.add_subplot(111)
            self.ax.text(0.5, 0.5, "No dataset loaded",
                        fontsize=16, ha='center', va='center',
                        transform=self.ax.transAxes, color='red')
            self.ax.set_title("No dataset loaded", fontsize=12)
            self.title_label.config(text="No dataset loaded")
            self.canvas.draw()
            return
        
        timestamps = self.current_dataset['timestamps']
        values = self.current_dataset['values']
        
        dataset_status = ""
        if self.current_data_id in self.test_datasets:
            dataset_status = " (TEST)"
            title_color = 'red'
        elif self.current_data_id in self.val_datasets:
            dataset_status = " (VAL)"
            title_color = 'orange'
        else:
            title_color = 'black'
        
        show_predictions = (self.current_dataset['predictions'] is not None and 
                           not self.stepwise_predicting)
        
        base_title = f"Dataset {self.current_data_id}{dataset_status} - {self.current_dataset['file']}"
        
        if self.view_mode == "2D":
            if self.show_overview:
                axes = []
                for i, feature in enumerate(self.features_info, 1):
                    ax = self.fig.add_subplot(3, 2, i)
                    axes.append(ax)
                    
                    feature_index = feature["index"]
                    feature_data = values[:, feature_index]
                    
                    if len(feature_data) > 0:
                        min_val = np.min(feature_data)
                        max_val = np.max(feature_data)
                        mean_val = np.mean(feature_data)
                        std_val = np.std(feature_data)
                    else:
                        min_val = max_val = mean_val = std_val = 0

                    if abs(mean_val)<1e-4 and std_val<1e-4:
                        ax.set_ylim(-1e-4,1e-4)
                    
                    ax.plot(timestamps, feature_data, 
                            color=feature["color"], 
                            linewidth=0.8, 
                            alpha=0.8, 
                            label='Original')
                    
                    if show_predictions:
                        pred_timestamps = self.current_dataset['predictions']['timestamps']
                        pred_values = self.current_dataset['predictions']['values']
                        pred_idx_raw = self.current_dataset['predictions'].get('points_idx')

                        min_len = min(len(pred_timestamps), len(pred_values))
                        if min_len > 0 and i <= pred_values.shape[1]:
                            pred_index = i - 1
                            if pred_index < pred_values.shape[1]:
                                # pred_values 已是绝对状态（增量 Δ 已在推理侧加回窗口末状态）
                                x_axis = (timestamps[pred_idx_raw[:min_len]]
                                          if pred_idx_raw is not None
                                          else pred_timestamps[:min_len])
                                ax.plot(x_axis,
                                    pred_values[:min_len, pred_index],
                                    'ro', markersize=1.5, alpha=0.6,
                                    label=f'Pred (t+{self.pre_time})')
                    
                    ax.set_title(f"{feature['name']}", fontsize=9, pad=3)
                    ax.set_ylabel(f"{feature['unit']}", fontsize=8)
                    ax.grid(True, linestyle='--', alpha=0.3)
                    ax.tick_params(axis='both', which='major', labelsize=7)
                    
                    if show_predictions and min_len > 0:
                        stats_text = (f"Min: {min_val:.10f} \n"
                                f"Max: {max_val:.10f} \n"
                                # f"max error: {error:.10f} \n"
                                )
                    else:
                        stats_text = ""
                    
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                            fontsize=15, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    if show_predictions and min_len > 0:
                        ax.legend(loc='best', fontsize=8)
                
                axes[-1].set_xlabel("Timestamp (s)", fontsize=9)
                axes[-2].set_xlabel("Timestamp (s)", fontsize=9)
                
                self.fig.subplots_adjust(
                    left=0.07,
                    right=0.98,
                    bottom=0.08,
                    top=0.92,
                    wspace=0.25,
                    hspace=0.4
                )
                
                self.update_chart_title_with_progress()
                
            else:
                selected_feature = self.current_feature
                
                self.ax = self.fig.add_subplot(111)
                
                feature_index = selected_feature["index"]
                feature_data = values[:, feature_index]
                
                if len(feature_data) > 0:
                    min_val = np.min(feature_data)
                    max_val = np.max(feature_data)
                    mean_val = np.mean(feature_data)
                    std_val = np.std(feature_data)
                    median_val = np.median(feature_data)
                    data_points = len(feature_data)
                else:
                    min_val = max_val = mean_val = std_val = median_val = 0
                    data_points = 0
                
                self.ax.plot(values[:,3], values[:,4], 
                            color=selected_feature["color"], 
                            linewidth=2.0, 
                            alpha=0.8, 
                            label=f'Original {selected_feature["name"]}')
                
                if show_predictions:
                    pred_timestamps = self.current_dataset['predictions']['timestamps']
                    pred_values = self.current_dataset['predictions']['values']
                    pred_idx_raw = self.current_dataset['predictions'].get('points_idx')

                    min_len = min(len(pred_timestamps), len(pred_values))
                    if min_len > 0:
                        pred_index = self.features_info.index(selected_feature)
                        if pred_index < pred_values.shape[1]:
                            x_axis = (timestamps[pred_idx_raw[:min_len]]
                                      if pred_idx_raw is not None
                                      else pred_timestamps[:min_len])
                            self.ax.plot(x_axis,
                                    pred_values[:min_len, pred_index],
                                    'ro', markersize=5, alpha=0.7,
                                    label=f'Prediction (t+{self.pre_time})')
                
                self.ax.set_xlabel("Timestamp (s)", fontsize=12)
                self.ax.set_ylabel(f"{selected_feature['name']} ({selected_feature['unit']})", fontsize=12)
                self.ax.grid(True, linestyle='--', alpha=0.6)
                
                if show_predictions and min_len > 0:
                    self.ax.legend(loc='best', fontsize=10)
                
                stats_text = (f"Min: {min_val:.4f} {selected_feature['unit']}\n"
                            f"Max: {max_val:.4f} {selected_feature['unit']}\n"
                            f"Mean: {mean_val:.4f} {selected_feature['unit']}\n"
                            f"Std Dev: {std_val:.4f} {selected_feature['unit']}\n"
                            f"Median: {median_val:.4f} {selected_feature['unit']}\n"
                            f"Data Points: {data_points}")
                if show_predictions and min_len > 0:
                    stats_text = (f"Min: {min_val:.4f} {selected_feature['unit']}\n"
                            f"Max: {max_val:.4f} {selected_feature['unit']}\n"
                            # f"Mean: {error:.4f} {selected_feature['unit']}\n"
                            f"Median: {median_val:.4f} {selected_feature['unit']}\n"
                            f"Data Points: {data_points}")
                
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
                self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes,
                            fontsize=10, verticalalignment='top', 
                            bbox=props)
                
                self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
                
                self.fig.subplots_adjust(
                    left=0.1,
                    right=0.95,
                    bottom=0.1,
                    top=0.88
                )
                
                self.update_chart_title_with_progress()
        
        elif self.view_mode == "3D":
            # 创建3D坐标系
            self.ax = self.fig.add_subplot(111, projection='3d')
            
            # 清理旧的预测轨迹
            if hasattr(self, 'prediction_line') and self.prediction_line is not None:
                try:
                    if hasattr(self.prediction_line, 'remove'):
                        self.prediction_line.remove()
                except:
                    pass
                self.prediction_line = None
            
            # 清理旧的轨迹线
            if hasattr(self, 'trajectory_line') and self.trajectory_line is not None:
                try:
                    if hasattr(self.trajectory_line, 'remove'):
                        self.trajectory_line.remove()
                except:
                    pass
                self.trajectory_line = None
            
            # 提取轨迹数据
            x_data = values[:, 3]  # X坐标
            y_data = values[:, 4]  # Y坐标
            z_data = values[:, 5]  # Z坐标
            
                        
            # 绘制原始轨迹
            self.trajectory_line = self.ax.plot(x_data, y_data, z_data, 
                                               'b-', linewidth=2, alpha=0.7, 
                                               label='Original Trajectory')[0]
            
            # 绘制预测轨迹
            if show_predictions:
                pred_values = self.current_dataset['predictions']['values']
                if len(pred_values) > 0:
                    pred_x = pred_values[:, 0]
                    pred_y = pred_values[:, 1]
                    pred_z = pred_values[:, 2]
                    self.prediction_line = self.ax.plot(pred_x, pred_y, pred_z, 
                                                       'r-', linewidth=2, alpha=0.7, 
                                                       label='Predicted Trajectory')[0]
                
            # 创建初始水滴模型（从原点开始）
            initial_x = x_data[0]
            initial_y = y_data[0]
            initial_z = z_data[0]
            initial_pitch = values[0, 7]
            initial_yaw = values[0, 8]
            
            self.create_vehicle_model(initial_x, initial_y, initial_z, initial_pitch, initial_yaw)
            
            # 在原点添加一个标记点
            self.ax.scatter([0], [0], [0], c='red', s=50, marker='o', label='Origin (0,0,0)')
            
            # 设置3D图表属性
            self.ax.set_xlabel('X (m, relative to start)', fontsize=12)
            self.ax.set_ylabel('Y (m, relative to start)', fontsize=12)
            self.ax.set_zlabel('Z (m, relative to start)', fontsize=12)
            self.ax.set_title('3D State Trajectory', fontsize=14)
            
            # 添加网格
            self.ax.grid(True, linestyle='--', alpha=0.6)
            
            # 添加坐标轴
            self.ax.legend(loc='best')
            
            # 设置等比例缩放
            self.ax.set_aspect('auto')
            
            # 添加网格
            self.ax.grid(True, linestyle='--', alpha=0.6)
            
            self.fig.subplots_adjust(
                left=0.05,
                right=0.95,
                bottom=0.05,
                top=0.95
            )
            
            # 更新标题
            self.update_chart_title_with_progress()
        
        self.canvas.draw()

    def export_chart(self):
        """导出图表为图像"""
        if self.current_dataset is None:
            messagebox.showwarning("Warning", "No dataset loaded to export")
            return

        save_dir = filedialog.askdirectory(title="Select Save Directory")
        if not save_dir:
            return

        try:
            if self.view_mode == "2D":
                if self.show_overview:
                    filename = f"{self.current_dataset['file'].replace('.mat', '')}_overview.png"
                else:
                    filename = f"{self.current_dataset['file'].replace('.mat', '')}_{self.feature_var.get().replace(' ', '_')}.png"
            else:
                filename = f"{self.current_dataset['file'].replace('.mat', '')}_3d_trajectory.png"
            
            save_path = os.path.join(save_dir, filename)
            
            self.fig.savefig(save_path, dpi=300, bbox_inches='tight')

            self.status_var.set(f"Chart saved to: {save_path}")
            messagebox.showinfo("Export Successful", f"Chart successfully exported to:\n{save_path}")

        except Exception as e:
            self.status_var.set(f"Export error: {str(e)}")
            messagebox.showerror("Export Error", f"Error exporting chart:\n{str(e)}")


def build_argparser() -> "argparse.ArgumentParser":
    import argparse

    ap = argparse.ArgumentParser(prog="tsforecast-gui",
                                 description="飞行器时序状态预测 - 交互式可视化（Tkinter）")
    ap.add_argument("--data_dir", type=str, default=DATA_DIR,
                    help="数据根目录（递归扫描其下所有 .mat / .csv / .npy）")
    ap.add_argument("--model_dir", type=str, default=MODEL_DIR,
                    help="模型目录（其下每个子目录为一个训练产物：config.json/norm.json/checkpoint.pth）")
    ap.add_argument("--datasets", type=str, default=None,
                    help="只显示数据根目录下的这些子目录，逗号分隔；留空=全部")
    ap.add_argument("--input_len", type=int, default=None, help="覆盖窗口输入长度")
    ap.add_argument("--pred_len", type=int, default=None, help="覆盖预测步数")
    ap.add_argument("--data_step", type=int, default=None, help="覆盖采样间隔")
    return ap


def main(argv=None):
    import argparse  # noqa: F401  (build_argparser 内部按需导入)

    a = build_argparser().parse_args(argv)
    datasets = ([s for s in a.datasets.replace(";", ",").split(",") if s.strip()]
                if a.datasets else None)

    root = tk.Tk()
    SingleChartViewer(
        master=root,
        data_dir=a.data_dir,
        model_dir=a.model_dir,
        special_datasets_dir=a.model_dir,
        datasets=datasets,
        input_len=a.input_len,
        pred_len=a.pred_len,
        data_step=a.data_step,
    )
    root.mainloop()


if __name__ == "__main__":
    main()