# MPPI-HC: Hierarchical Compensated MPPI Controller

## Overview

MPPI-HC is a Model Predictive Path Integral controller with **hierarchical slip compensation** for omnidirectional swerve drive robots. It builds upon traditional MPPI with three key innovations:

### Hierarchical Architecture

```
Layer 1: MPPI Planning Layer
    ├── Samples trajectories in (vx, vy, ω) space
    ├── Uses slip-aware dynamics for prediction
    └── Evaluates with curvature-aware cost function

Layer 2: Slip Estimation Layer (Online Learning)
    ├── Gradient descent learning of slip factor K_slip
    ├── Low-pass filtering for noise rejection
    └── Convergence detection

Layer 3: Slip Compensation Layer (Feedforward)
    ├── Δvy = -γ · K_slip · vx · ω
    └── Active slip cancellation before output
```

## Key Features

### 1. Online Slip Estimation
- Uses gradient descent to learn slip factor in real-time
- Slip model: `v_slip = -K_slip · vx · ω`
- Adapts to changing surface conditions

### 2. Curvature-Aware Speed Regulation
- Computes maximum safe cornering speed: `v_safe = sqrt(μ·g / κ_eff)`
- Omnidirectional adaptation with curvature floor
- Proactive deceleration before corners

### 3. Yaw Rate Tracking (for Omnidirectional Robots)
- Uses heading error-based yaw rate control instead of path curvature
- Avoids unnecessary "swing out" before turns
- `ω_desired = -k · (yaw - yaw_ref)`

### 4. Feedforward Slip Compensation
- Predicts lateral slip and preemptively compensates
- Compensation gain γ ∈ [0, 1] for tuning

## Usage

### Launch

```bash
# Default configuration
roslaunch mppi_hc mppi_hc.launch

# Low friction configuration (μ = 0.3)
roslaunch mppi_hc mppi_hc.launch config_file:=$(rospack find mppi_hc)/config/mppi_hc_low_friction.yaml
```

### Topics

**Subscriptions:**
- `/odom` (nav_msgs/Odometry): Robot odometry
- `/reference_path` (nav_msgs/Path): Reference trajectory
- `/collision_costmap` (grid_map_msgs/GridMap): Collision costs
- `/distance_error_map` (grid_map_msgs/GridMap): Distance to reference path
- `/reference_yaw_map` (grid_map_msgs/GridMap): Reference heading map

**Publications:**
- `/cmd_vel` (geometry_msgs/Twist): Velocity command
- `/optimal_trajectory` (nav_msgs/Path): Planned trajectory
- `/mppi_hc_status` (jsk_rviz_plugins/OverlayText): Status display

## Configuration

Key parameters in `config/mppi_hc.yaml`:

```yaml
# Slip estimation
slip:
  learning_rate: 0.01       # Adaptation speed
  slip_factor_max: 0.3      # Maximum slip estimate
  compensation_gain: 0.7    # Feedforward gain γ

# Cost weights
cost:
  slip_risk: 15.0           # Penalize high-slip inputs
  curvature_speed: 60.0     # Curvature speed regulation
  yaw_rate_tracking: 25.0   # Yaw rate tracking for turns
```

## Academic Contribution

This controller integrates:
1. **Slip-aware dynamics** in MPPI rollouts
2. **Online adaptive estimation** of slip parameters
3. **Curvature-aware speed regulation** with omnidirectional adaptation
4. **Feedforward compensation** for predictive slip cancellation

These form a coherent hierarchy that improves rectangular path tracking on low-friction surfaces.

## Dependencies

- ROS Noetic
- grid_map
- Eigen3
- OpenMP
- mppi_eval_msgs

## Build

```bash
cd ~/catkin_ws
catkin build mppi_hc
```

## License

MIT License
