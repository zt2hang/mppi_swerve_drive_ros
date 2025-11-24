#!/home/zzt/planner_code/mppi_swerve_drive_ros/mppi_mlp_venv/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import time

class ResidualNetwork(nn.Module):
    """
    简单的 MLP 网络结构
    输入: 11维 (3维当前速度 + 8维控制量)
    输出: 3维 (速度残差: delta_vx, delta_vy, delta_omega)
    """
    def __init__(self, input_dim=11, output_dim=3, hidden_dim=32):
        super(ResidualNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(), # 或者 nn.Tanh()
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 关键: 初始化最后一层的权重为极小值，确保初始输出接近 0
        # 这样在训练初期，机器人的行为完全由运动学模型主导，保证安全
        self._init_last_layer()

    def _init_last_layer(self):
        # 获取最后一层 (Linear)
        last_layer = self.net[-1]
        # 将权重初始化为非常小的均匀分布
        nn.init.uniform_(last_layer.weight, -1e-6, 1e-6)
        # 将偏置初始化为 0
        nn.init.zeros_(last_layer.bias)

    def forward(self, x):
        return self.net(x)

class OnlineResidualLearner:
    def __init__(self, 
                 input_dim=11, 
                 output_dim=3, 
                 hidden_dim=32, 
                 buffer_size=5000, 
                 batch_size=64, 
                 lr=1e-3, 
                 weight_decay=1e-4,
                 device=None):
        """
        在线残差学习器
        
        Args:
            input_dim: 输入维度 (vx, vy, w, 4*wheel_vel, 4*steer_angle)
            output_dim: 输出维度 (delta_vx, delta_vy, delta_w)
            hidden_dim: 隐藏层神经元数量
            buffer_size: 经验回放缓冲区大小 (滑动窗口)
            batch_size: 每次训练采样的批次大小
            lr: 学习率
            weight_decay: L2 正则化系数
            device: 'cuda' or 'cpu', 如果为 None 则自动检测
        """
        
        # 设备配置
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"[OnlineResidualLearner] Initializing on device: {self.device}")

        # 初始化网络
        self.model = ResidualNetwork(input_dim, output_dim, hidden_dim).to(self.device)
        
        # 优化器 (Adam + L2 Regularization)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 经验回放缓冲区 (使用 deque 实现滑动窗口)
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
        # 数据归一化参数 (基于物理限制的固定归一化通常比在线统计更稳定)
        # 假设: Vel ~ [-2, 2], WheelVel ~ [-20, 20], Steer ~ [-pi, pi]
        # 这里我们使用简单的缩放因子，实际应用中可根据机器人参数调整
        self.input_scale = torch.tensor([
            1.0, 1.0, 1.0,              # Body Vel (vx, vy, w) - 假设量级在 1 左右
            0.1, 0.1, 0.1, 0.1,         # Wheel Vel - 假设最大 10-20 m/s，缩放 0.1
            1.0/np.pi, 1.0/np.pi, 1.0/np.pi, 1.0/np.pi # Steer Angle - 归一化到 [-1, 1]
        ], device=self.device, dtype=torch.float32)
        
        # 输出截断阈值 (m/s, rad/s) - 防止网络输出过大的残差导致不稳定
        # Reduced from 0.5 to 0.2 for stability
        self.output_clamp = 0.2 

    def normalize_input(self, x):
        """ 输入归一化 """
        return x * self.input_scale

    def add_observation(self, current_vel, control_cmd, target_residual):
        """
        添加一条观测数据到缓冲区
        
        Args:
            current_vel: np.array or list [vx, vy, w]
            control_cmd: np.array or list [v_fl, v_fr, v_rl, v_rr, th_fl, th_fr, th_rl, th_rr]
            target_residual: np.array or list [delta_vx, delta_vy, delta_w] 
                             (计算方式: Real_Next_Vel - Kinematic_Next_Vel)
        """
        # 拼接状态和动作作为输入
        state_action = np.concatenate([current_vel, control_cmd])
        
        # 存入缓冲区 (保持为 numpy 格式以节省显存，训练时再转 Tensor)
        self.replay_buffer.append((state_action, target_residual))

    def train_step(self):
        """
        执行一步训练
        从缓冲区采样 -> 前向传播 -> 计算 Loss -> 反向传播 -> 更新权重
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0 # 数据不够，不训练

        # 1. 随机采样
        batch = random.sample(self.replay_buffer, self.batch_size)
        state_actions, targets = zip(*batch)
        
        # 2. 转换为 Tensor 并移至 GPU
        inputs = torch.tensor(np.array(state_actions), dtype=torch.float32).to(self.device)
        targets = torch.tensor(np.array(targets), dtype=torch.float32).to(self.device)
        
        # 3. 归一化输入
        inputs_norm = self.normalize_input(inputs)
        
        # 4. 训练模式
        self.model.train()
        
        # 5. 前向传播
        predictions = self.model(inputs_norm)
        
        # 6. 计算损失
        loss = self.criterion(predictions, targets)
        
        # 7. 反向传播与优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def predict_batch(self, state_batch, action_batch):
        """
        MPPI 批量推理接口 (高性能)
        
        Args:
            state_batch: Tensor (K, 3) [vx, vy, w]
            action_batch: Tensor (K, 8) [wheel_vels..., steer_angles...]
            
        Returns:
            residuals: Tensor (K, 3) [delta_vx, delta_vy, delta_w]
        """
        # 确保输入是 Tensor 且在正确的设备上
        if not isinstance(state_batch, torch.Tensor):
            state_batch = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
        if not isinstance(action_batch, torch.Tensor):
            action_batch = torch.tensor(action_batch, dtype=torch.float32, device=self.device)
            
        # 拼接输入 (K, 11)
        inputs = torch.cat([state_batch, action_batch], dim=1)
        
        # 归一化
        inputs_norm = self.normalize_input(inputs)
        
        # 推理模式 (关闭 Dropout/BatchNorm 更新，不计算梯度)
        self.model.eval()
        with torch.no_grad():
            residuals = self.model(inputs_norm)
            
        # 输出截断 (Clamping) - 保证安全性
        residuals = torch.clamp(residuals, -self.output_clamp, self.output_clamp)
        
        return residuals

# ==========================================
# Dummy Usage Example
# ==========================================
if __name__ == "__main__":
    print("=== Testing OnlineResidualLearner ===")
    
    # 1. 初始化
    learner = OnlineResidualLearner(
        input_dim=11, 
        output_dim=3, 
        hidden_dim=32, 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 2. 模拟数据收集 (Add Observations)
    print("Collecting dummy data...")
    for _ in range(100):
        # 模拟当前状态 [vx, vy, w]
        curr_vel = np.random.uniform(-1.0, 1.0, 3)
        # 模拟控制指令 [4*vel, 4*steer]
        cmd = np.random.uniform(-5.0, 5.0, 8)
        # 模拟目标残差 (真实值 - 运动学预测值)
        # 假设真实残差很小
        target_res = np.random.normal(0, 0.05, 3)
        
        learner.add_observation(curr_vel, cmd, target_res)
        
    # 3. 训练 (Train Step)
    print("Training...")
    for i in range(5):
        loss = learner.train_step()
        print(f"Step {i+1}, Loss: {loss:.6f}")
        
    # 4. MPPI 批量预测 (Batch Prediction)
    print("Running Batch Prediction (MPPI style)...")
    K = 3000 # MPPI 采样数
    
    # 模拟 MPPI 生成的随机状态和动作批次
    # 注意：在实际 MPPI 中，这些通常已经在 GPU 上了
    dummy_states = torch.randn(K, 3).to(learner.device)
    dummy_actions = torch.randn(K, 8).to(learner.device)
    
    start_time = time.time()
    residuals = learner.predict_batch(dummy_states, dummy_actions)
    end_time = time.time()
    
    print(f"Prediction Shape: {residuals.shape}")
    print(f"Inference Time for {K} samples: {(end_time - start_time)*1000:.3f} ms")
    print(f"Sample Residuals (First 3):\n{residuals[:3].cpu().numpy()}")
    
    # 验证初始输出是否接近 0 (因为我们做了特殊初始化)
    mean_res = residuals.abs().mean().item()
    print(f"Mean Absolute Residual: {mean_res:.6f} (Should be small if untrained or trained on small targets)")
