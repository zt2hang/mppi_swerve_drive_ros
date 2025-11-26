#pragma once

/**
 * @file types.hpp
 * @brief Common type definitions for MPPI-HC controller
 * 
 * This file defines the fundamental data types used throughout the
 * Hierarchical Compensated MPPI controller.
 */

#include <Eigen/Dense>
#include <vector>
#include <array>
#include <cmath>

namespace mppi_hc
{

// ============================================================================
// Basic State and Control Types
// ============================================================================

/**
 * @brief 2D pose (x, y, yaw)
 */
struct Pose2D
{
    double x = 0.0;      // [m]
    double y = 0.0;      // [m]
    double yaw = 0.0;    // [rad]

    Eigen::Vector3d toEigen() const {
        return Eigen::Vector3d(x, y, yaw);
    }

    void fromEigen(const Eigen::Vector3d& v) {
        x = v(0); y = v(1); yaw = v(2);
    }

    // Normalize yaw to [-pi, pi]
    void normalizeYaw() {
        yaw = std::remainder(yaw, 2.0 * M_PI);
    }
};

/**
 * @brief Full state with pose and velocities
 */
struct FullState
{
    double x = 0.0;      // Position x [m]
    double y = 0.0;      // Position y [m]
    double yaw = 0.0;    // Heading [rad]
    double vx = 0.0;     // Forward velocity [m/s]
    double vy = 0.0;     // Lateral velocity [m/s]
    double omega = 0.0;  // Angular velocity [rad/s]

    Pose2D pose() const {
        return Pose2D{x, y, yaw};
    }

    void normalizeYaw() {
        yaw = std::remainder(yaw, 2.0 * M_PI);
    }

    double speed() const {
        return std::sqrt(vx * vx + vy * vy);
    }
};

/**
 * @brief Body velocity command (vx, vy, omega)
 */
struct BodyVelocity
{
    double vx = 0.0;     // Forward velocity [m/s]
    double vy = 0.0;     // Lateral velocity [m/s]
    double omega = 0.0;  // Angular velocity [rad/s]

    Eigen::Vector3d toEigen() const {
        return Eigen::Vector3d(vx, vy, omega);
    }

    void fromEigen(const Eigen::Vector3d& v) {
        vx = v(0); vy = v(1); omega = v(2);
    }

    double speed() const {
        return std::sqrt(vx * vx + vy * vy);
    }

    void setZero() {
        vx = 0.0; vy = 0.0; omega = 0.0;
    }

    // Clamp velocities to limits
    void clamp(double vx_max, double vy_max, double omega_max) {
        vx = std::clamp(vx, -vx_max, vx_max);
        vy = std::clamp(vy, -vy_max, vy_max);
        omega = std::clamp(omega, -omega_max, omega_max);
    }
};

/**
 * @brief Single wheel command (velocity and steering angle)
 */
struct WheelCommand
{
    double velocity = 0.0;  // Wheel velocity [m/s]
    double steer = 0.0;     // Steering angle [rad]
};

/**
 * @brief Full 8-DOF vehicle command (4 wheels x 2 DOF each)
 */
struct VehicleCommand8D
{
    WheelCommand fl;  // Front-left
    WheelCommand fr;  // Front-right
    WheelCommand rl;  // Rear-left
    WheelCommand rr;  // Rear-right

    Eigen::Matrix<double, 8, 1> toEigen() const {
        Eigen::Matrix<double, 8, 1> v;
        v << fl.steer, fr.steer, rl.steer, rr.steer,
             fl.velocity, fr.velocity, rl.velocity, rr.velocity;
        return v;
    }
};

// ============================================================================
// Vehicle Parameters
// ============================================================================

/**
 * @brief Vehicle geometry and physical parameters
 */
struct VehicleParams
{
    // Geometry
    double l_f = 0.5;         // Front axle to CoG [m]
    double l_r = 0.5;         // Rear axle to CoG [m]
    double d_l = 0.5;         // Left wheel to CoG [m]
    double d_r = 0.5;         // Right wheel to CoG [m]
    double tire_radius = 0.2; // Tire radius [m]

    // Velocity limits
    double vx_max = 3.0;      // Max forward velocity [m/s]
    double vy_max = 3.0;      // Max lateral velocity [m/s]
    double omega_max = 3.0;   // Max angular velocity [rad/s]

    // Physical limits
    double max_steer_angle = M_PI / 2;  // Max steering angle [rad]
    double max_wheel_vel = 10.0;        // Max wheel velocity [rad/s]

    // Wheelbase and track
    double wheelbase() const { return l_f + l_r; }
    double track() const { return d_l + d_r; }
};

// ============================================================================
// MPPI Controller Parameters
// ============================================================================

/**
 * @brief MPPI sampling and optimization parameters
 */
struct MPPIParams
{
    // Sampling
    int num_samples = 3000;        // Number of trajectory samples (K)
    int prediction_horizon = 50;   // Prediction steps (T)
    double step_dt = 0.033;        // Time step [s]
    double exploration_ratio = 0.1; // Ratio of pure exploration samples

    // MPPI algorithm parameters
    double lambda = 100.0;         // Temperature parameter
    double alpha = 0.975;          // Control smoothing factor

    // Noise standard deviations
    Eigen::Vector3d sigma = Eigen::Vector3d(0.5, 0.5, 0.8);  // [vx, vy, omega]

    // Reference velocity
    double ref_velocity = 2.0;     // [m/s]
};

/**
 * @brief Cost function weights
 */
struct CostWeights
{
    // Path tracking
    double distance_error = 40.0;
    double angular_error = 30.0;
    double velocity_error = 10.0;
    double terminal_state = 10.0;

    // Safety
    double collision_penalty = 50.0;

    // Slip-aware costs
    double slip_risk = 15.0;              // Penalize high slip risk inputs
    double curvature_speed = 60.0;        // Penalize exceeding safe cornering speed
    double yaw_rate_tracking = 25.0;      // Yaw rate tracking for turns

    // Smoothness
    std::array<double, 3> cmd_change = {0.0, 0.0, 0.0};  // [vx, vy, omega]
    std::array<double, 8> vehicle_cmd_change = {1.4, 1.4, 1.4, 1.4, 0.1, 0.1, 0.1, 0.1};
};

/**
 * @brief Slip estimation and compensation parameters
 */
struct SlipParams
{
    // Estimator
    double learning_rate = 0.01;
    double slip_factor_min = 0.0;
    double slip_factor_max = 0.3;
    double excitation_threshold = 0.1;  // Min |vx * omega| for learning

    // Curvature-aware speed regulation
    double base_friction_coeff = 0.3;
    double curvature_lookahead = 0.5;   // [m]
    double curvature_floor = 0.5;       // Min effective curvature for omni robots
    double speed_margin = 0.3;          // Speed tolerance ratio

    // Feedforward compensation
    bool enable_compensation = true;
    double compensation_gain = 0.7;     // Feedforward gain
};

/**
 * @brief Complete controller configuration
 */
struct ControllerConfig
{
    VehicleParams vehicle;
    MPPIParams mppi;
    CostWeights weights;
    SlipParams slip;

    // Navigation tolerances
    double xy_goal_tolerance = 0.5;   // [m]
    double yaw_goal_tolerance = 0.5;  // [rad]

    // Savitzky-Golay filter
    bool use_sg_filter = true;
    int sg_half_window = 4;
    int sg_poly_order = 6;
};

// ============================================================================
// Type Aliases for Collections
// ============================================================================

using State = FullState;       // Full state with velocities
using PoseState = Pose2D;      // Pose-only state for goal specification
using Control = BodyVelocity;
using StateSequence = std::vector<State>;
using ControlSequence = std::vector<Control>;
using StateTrajectories = std::vector<StateSequence>;
using ControlTrajectories = std::vector<ControlSequence>;
using CostVector = std::vector<double>;

} // namespace mppi_hc
