# MPPI-HC: 分层补偿 MPPI 控制器

## 1. 简介 (Introduction)

**MPPI-HC** (Hierarchical Compensated Model Predictive Path Integral) 是一个专为在变摩擦或低摩擦表面上运行的全向舵轮驱动机器人设计的鲁棒控制框架。通过将滑移感知动力学模型与在线参数估计相结合，MPPI-HC 解决了纯运动学控制器在高滑移条件下的局限性。

该系统采用分层架构，能够同时进行最优轨迹规划、环境参数估计以及未建模动力学的补偿。

## 2. 系统架构 (System Architecture)

控制器由三个相互作用的层级组成：

### 第一层：规划层 (Slip-Aware MPPI)
主层利用模型预测路径积分 (MPPI) 控制来采样数千条潜在轨迹。与标准 MPPI 不同，它采用**滑移感知动力学模型**来预测未来状态。该模型显式地考虑了纵向速度、横摆角速度与由此产生的侧向滑移之间的耦合关系，使规划器能够“预见”并规避高滑移风险的动作。

### 第二层：估计层 (Online Learning)
在线估计器并行运行以识别滑移系数 $K_{slip}$。它使用梯度下降法最小化模型预测状态与实际机器人状态之间的预测误差。这使得控制器能够实时适应不断变化的表面摩擦条件（例如从地毯移动到瓷砖），而无需人工调整。

### 第三层：补偿层 (Feedforward)
最终的控制输出由前馈补偿器进行调节。基于规划的指令和估计的滑移因子，该层注入一个侧向速度分量以主动抵消预测的滑移，确保机器人在显著滑移条件下也能精准跟踪预定路径。

## 3. 数学公式 (Mathematical Formulation)

### 3.1. 滑移感知动力学模型
标准运动学模型假设没有侧滑。MPPI-HC 增加了一个源自机器人向心加速度的滑移项。有效侧向速度 $v_{y,eff}$ 建模为：

$$ v_{y,eff} = v_y - K_{slip} \cdot v_x \cdot \omega $$

其中：
- $v_x, v_y$: 指令体坐标系速度
- $\omega$: 横摆角速度
- $K_{slip}$: 可学习的滑移因子

状态演变方程为：
$$ \dot{x} = v_x \cos(\theta) - v_{y,eff} \sin(\theta) $$
$$ \dot{y} = v_x \sin(\theta) + v_{y,eff} \cos(\theta) $$
$$ \dot{\theta} = \omega $$

### 3.2. 代价函数
MPPI 优化最小化包含跟踪精度、稳定性和滑移风险的代价函数 $J$：

$$ J(x, u) = w_{track} \|x - x_{ref}\|^2 + w_{slip} J_{slip} + w_{curve} J_{curve} $$

*   **滑移风险代价 ($J_{slip}$)**: 惩罚预计会引起高滑移速度的控制动作。
    $$ J_{slip} = (K_{slip} \cdot v_x \cdot \omega)^2 $$

*   **曲率感知速度限制 ($J_{curve}$)**: 基于瞬时路径曲率 $\kappa$ 和摩擦系数 $\mu$ 施加动力学限制。
    $$ v_{limit} = \sqrt{\frac{\mu g}{\kappa}} $$
    如果机器人速度超过此限制，代价将呈指数级增加。

## 4. 关键特性 (Key Features)

1.  **在线滑移估计**: 使用梯度下降实时学习 $K_{slip}$，适应表面条件。
2.  **曲率感知速度调节**: 基于估计的摩擦圆，在弯道前主动减速。
3.  **横摆角跟踪**: 独立于路径切线优化航向，充分利用舵轮驱动的全向能力。
4.  **前馈滑移补偿**: 在最终指令中加入 $\Delta v_y = -\gamma \cdot K_{slip} \cdot v_x \cdot \omega$ 以抵消滑移。

## 5. 使用与配置 (Usage & Configuration)

### 启动 (Launch)
```bash
# 默认配置
roslaunch mppi_hc mppi_hc.launch

# 低摩擦配置 (μ = 0.3)
roslaunch mppi_hc mppi_hc.launch config_file:=$(rospack find mppi_hc)/config/mppi_hc_low_friction.yaml
```

### 关键参数 (`config/mppi_hc.yaml`)
```yaml
slip:
  learning_rate: 0.01       # K_slip 的适应速度
  slip_factor_max: 0.3      # 滑移因子的上限
  compensation_gain: 0.7    # 前馈增益 (0.0 到 1.0)

cost:
  slip_risk: 15.0           # 滑移风险惩罚权重
  curvature_speed: 60.0     # 速度调节权重
```

## 6. 依赖 (Dependencies)
- ROS Noetic
- grid_map
- Eigen3
- OpenMP
