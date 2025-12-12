#pragma once

/**
 * @file types.hpp
 * @brief Core type definitions for MPPI-TireForce controller
 * 
 * This file defines the fundamental data structures used throughout the
 * tire force-based MPPI controller, including:
 * - State representations (pose, velocity, full state)
 * - Control inputs (body forces, wheel commands)
 * - Vehicle and tire parameters
 * - Controller configuration
 * 
 * @author ZZT
 * @date 2024
 */

#include <Eigen/Dense>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>

namespace mppi_tf
{

// ============================================================================
// Constants
// ============================================================================

constexpr double GRAVITY = 9.81;           // [m/s²]
constexpr double EPSILON = 1e-6;           // Numerical stability threshold
constexpr int NUM_WHEELS = 4;              // Number of wheels
constexpr int FORCE_DIM = 3;               // Force control dimension (Fx, Fy, Mz)
constexpr int STATE_DIM = 6;               // Full state dimension (x, y, θ, vx, vy, ω)
constexpr int POSE_DIM = 3;                // Pose dimension (x, y, θ)

// Wheel indices
enum WheelIndex : int {
    FRONT_LEFT = 0,
    FRONT_RIGHT = 1,
    REAR_LEFT = 2,
    REAR_RIGHT = 3
};

// ============================================================================
// Basic Geometric Types
// ============================================================================

/**
 * @brief 2D position
 */
struct Position2D
{
    double x = 0.0;  // [m]
    double y = 0.0;  // [m]

    Eigen::Vector2d toEigen() const { return Eigen::Vector2d(x, y); }
    double norm() const { return std::hypot(x, y); }
};

/**
 * @brief 2D pose (position + heading)
 */
struct Pose2D
{
    double x = 0.0;    // [m]
    double y = 0.0;    // [m]
    double yaw = 0.0;  // [rad]

    Eigen::Vector3d toEigen() const { return Eigen::Vector3d(x, y, yaw); }
    
    void fromEigen(const Eigen::Vector3d& v) {
        x = v(0); y = v(1); yaw = v(2);
    }

    void normalizeYaw() {
        yaw = std::remainder(yaw, 2.0 * M_PI);
    }

    double distanceTo(const Pose2D& other) const {
        return std::hypot(x - other.x, y - other.y);
    }
};

// ============================================================================
// Velocity Types
// ============================================================================

/**
 * @brief Body-frame velocity (vx, vy, omega)
 */
struct BodyVelocity
{
    double vx = 0.0;     // [m/s] Forward velocity
    double vy = 0.0;     // [m/s] Lateral velocity (positive = left)
    double omega = 0.0;  // [rad/s] Angular velocity (positive = CCW)

    Eigen::Vector3d toEigen() const { return Eigen::Vector3d(vx, vy, omega); }
    
    void fromEigen(const Eigen::Vector3d& v) {
        vx = v(0); vy = v(1); omega = v(2);
    }

    double speed() const { return std::hypot(vx, vy); }
    double heading() const { return std::atan2(vy, vx); }

    void setZero() { vx = 0.0; vy = 0.0; omega = 0.0; }

    void clamp(double vx_max, double vy_max, double omega_max) {
        vx = std::clamp(vx, -vx_max, vx_max);
        vy = std::clamp(vy, -vy_max, vy_max);
        omega = std::clamp(omega, -omega_max, omega_max);
    }

    BodyVelocity operator+(const BodyVelocity& other) const {
        return {vx + other.vx, vy + other.vy, omega + other.omega};
    }

    BodyVelocity operator*(double scale) const {
        return {vx * scale, vy * scale, omega * scale};
    }
};

// ============================================================================
// Full State
// ============================================================================

/**
 * @brief Full vehicle state (pose + velocity)
 */
struct FullState
{
    // Position and orientation (world frame)
    double x = 0.0;      // [m]
    double y = 0.0;      // [m]
    double yaw = 0.0;    // [rad]
    
    // Velocities (body frame)
    double vx = 0.0;     // [m/s]
    double vy = 0.0;     // [m/s]
    double omega = 0.0;  // [rad/s]

    Pose2D pose() const { return Pose2D{x, y, yaw}; }
    BodyVelocity velocity() const { return BodyVelocity{vx, vy, omega}; }
    double speed() const { return std::hypot(vx, vy); }

    void normalizeYaw() { yaw = std::remainder(yaw, 2.0 * M_PI); }

    Eigen::Matrix<double, 6, 1> toEigen() const {
        Eigen::Matrix<double, 6, 1> v;
        v << x, y, yaw, vx, vy, omega;
        return v;
    }

    void fromEigen(const Eigen::Matrix<double, 6, 1>& v) {
        x = v(0); y = v(1); yaw = v(2);
        vx = v(3); vy = v(4); omega = v(5);
    }

    void setPose(const Pose2D& p) { x = p.x; y = p.y; yaw = p.yaw; }
    void setVelocity(const BodyVelocity& v) { vx = v.vx; vy = v.vy; omega = v.omega; }
};

// ============================================================================
// Force Types
// ============================================================================

/**
 * @brief 2D force vector (for single wheel or total)
 */
struct Force2D
{
    double Fx = 0.0;  // [N] Longitudinal force (tire x-direction)
    double Fy = 0.0;  // [N] Lateral force (tire y-direction)

    Eigen::Vector2d toEigen() const { return Eigen::Vector2d(Fx, Fy); }
    
    void fromEigen(const Eigen::Vector2d& v) { Fx = v(0); Fy = v(1); }

    double magnitude() const { return std::hypot(Fx, Fy); }
    double direction() const { return std::atan2(Fy, Fx); }

    Force2D operator+(const Force2D& other) const {
        return {Fx + other.Fx, Fy + other.Fy};
    }

    Force2D operator*(double scale) const {
        return {Fx * scale, Fy * scale};
    }

    /**
     * @brief Scale force to fit within friction circle
     * @param max_force Maximum allowable force magnitude
     */
    void saturate(double max_force) {
        double mag = magnitude();
        if (mag > max_force && mag > EPSILON) {
            double scale = max_force / mag;
            Fx *= scale;
            Fy *= scale;
        }
    }
};

/**
 * @brief Total body forces and moment
 * This is the control input in force space
 */
struct BodyForce
{
    double Fx = 0.0;   // [N] Total longitudinal force (body x)
    double Fy = 0.0;   // [N] Total lateral force (body y)
    double Mz = 0.0;   // [Nm] Total yaw moment

    Eigen::Vector3d toEigen() const { return Eigen::Vector3d(Fx, Fy, Mz); }
    
    void fromEigen(const Eigen::Vector3d& v) {
        Fx = v(0); Fy = v(1); Mz = v(2);
    }

    void setZero() { Fx = 0.0; Fy = 0.0; Mz = 0.0; }

    void clamp(double Fx_max, double Fy_max, double Mz_max) {
        Fx = std::clamp(Fx, -Fx_max, Fx_max);
        Fy = std::clamp(Fy, -Fy_max, Fy_max);
        Mz = std::clamp(Mz, -Mz_max, Mz_max);
    }

    BodyForce operator+(const BodyForce& other) const {
        return {Fx + other.Fx, Fy + other.Fy, Mz + other.Mz};
    }

    BodyForce operator*(double scale) const {
        return {Fx * scale, Fy * scale, Mz * scale};
    }
};

// ============================================================================
// Wheel-Level Types
// ============================================================================

/**
 * @brief Single wheel state including slip information
 */
struct WheelState
{
    // Wheel position relative to CoG (body frame)
    double rx = 0.0;  // [m] x-offset from CoG
    double ry = 0.0;  // [m] y-offset from CoG
    
    // Wheel kinematics
    double vx_wheel = 0.0;   // [m/s] Wheel hub velocity in x
    double vy_wheel = 0.0;   // [m/s] Wheel hub velocity in y
    double wheel_speed = 0.0; // [rad/s] Wheel angular velocity
    double steer_angle = 0.0; // [rad] Steering angle

    // Slip variables
    double slip_angle = 0.0;  // [rad] Tire slip angle
    double slip_ratio = 0.0;  // [-] Longitudinal slip ratio
    
    // Forces
    double Fx = 0.0;  // [N] Tire longitudinal force
    double Fy = 0.0;  // [N] Tire lateral force
    double Fz = 0.0;  // [N] Tire normal load

    // Derived quantities
    double getFrictionUtilization(double mu) const {
        double max_force = mu * Fz;
        return (max_force > EPSILON) ? std::hypot(Fx, Fy) / max_force : 0.0;
    }
};

/**
 * @brief Single wheel command
 */
struct WheelCommand
{
    double velocity = 0.0;    // [m/s] Wheel velocity (linear)
    double steer_angle = 0.0; // [rad] Steering angle

    void clamp(double v_max, double steer_max) {
        velocity = std::clamp(velocity, -v_max, v_max);
        steer_angle = std::clamp(steer_angle, -steer_max, steer_max);
    }
};

/**
 * @brief Full 8-DOF vehicle command (4 wheels × 2 DOF)
 */
struct VehicleCommand8D
{
    std::array<WheelCommand, NUM_WHEELS> wheels;

    WheelCommand& fl() { return wheels[FRONT_LEFT]; }
    WheelCommand& fr() { return wheels[FRONT_RIGHT]; }
    WheelCommand& rl() { return wheels[REAR_LEFT]; }
    WheelCommand& rr() { return wheels[REAR_RIGHT]; }

    const WheelCommand& fl() const { return wheels[FRONT_LEFT]; }
    const WheelCommand& fr() const { return wheels[FRONT_RIGHT]; }
    const WheelCommand& rl() const { return wheels[REAR_LEFT]; }
    const WheelCommand& rr() const { return wheels[REAR_RIGHT]; }

    Eigen::Matrix<double, 8, 1> toEigen() const {
        Eigen::Matrix<double, 8, 1> v;
        v << wheels[0].steer_angle, wheels[1].steer_angle, 
             wheels[2].steer_angle, wheels[3].steer_angle,
             wheels[0].velocity, wheels[1].velocity,
             wheels[2].velocity, wheels[3].velocity;
        return v;
    }

    void clamp(double v_max, double steer_max) {
        for (auto& w : wheels) {
            w.clamp(v_max, steer_max);
        }
    }
};

/**
 * @brief Wheel forces for all four wheels
 */
struct WheelForces
{
    std::array<Force2D, NUM_WHEELS> forces;

    Force2D& fl() { return forces[FRONT_LEFT]; }
    Force2D& fr() { return forces[FRONT_RIGHT]; }
    Force2D& rl() { return forces[REAR_LEFT]; }
    Force2D& rr() { return forces[REAR_RIGHT]; }

    const Force2D& fl() const { return forces[FRONT_LEFT]; }
    const Force2D& fr() const { return forces[FRONT_RIGHT]; }
    const Force2D& rl() const { return forces[REAR_LEFT]; }
    const Force2D& rr() const { return forces[REAR_RIGHT]; }

    Force2D& operator[](int idx) { return forces[idx]; }
    const Force2D& operator[](int idx) const { return forces[idx]; }
};

// ============================================================================
// Vehicle Parameters
// ============================================================================

/**
 * @brief Vehicle geometry and mass properties
 */
struct VehicleGeometry
{
    // Distances from CoG to axles
    double l_f = 0.5;   // [m] CoG to front axle
    double l_r = 0.5;   // [m] CoG to rear axle
    
    // Distances from CoG to wheel centerlines
    double d_l = 0.5;   // [m] CoG to left wheels
    double d_r = 0.5;   // [m] CoG to right wheels

    // Wheel properties
    double tire_radius = 0.2;  // [m] Effective tire radius

    // Derived geometry
    double wheelbase() const { return l_f + l_r; }
    double track() const { return d_l + d_r; }

    /**
     * @brief Get wheel position in body frame
     */
    Position2D getWheelPosition(WheelIndex idx) const {
        Position2D pos;
        switch (idx) {
            case FRONT_LEFT:  pos.x = l_f;  pos.y = d_l;  break;
            case FRONT_RIGHT: pos.x = l_f;  pos.y = -d_r; break;
            case REAR_LEFT:   pos.x = -l_r; pos.y = d_l;  break;
            case REAR_RIGHT:  pos.x = -l_r; pos.y = -d_r; break;
        }
        return pos;
    }
};

/**
 * @brief Vehicle mass and inertia properties
 */
struct VehicleMass
{
    double mass = 100.0;     // [kg] Total vehicle mass
    double Iz = 50.0;        // [kg·m²] Yaw moment of inertia
    double h_cog = 0.3;      // [m] CoG height (for load transfer)

    // Static weight distribution (front/total)
    double weight_dist_front = 0.5;

    /**
     * @brief Get static normal load on each wheel
     */
    double getStaticWheelLoad(WheelIndex idx) const {
        double total_weight = mass * GRAVITY;
        double front_load = total_weight * weight_dist_front;
        double rear_load = total_weight * (1.0 - weight_dist_front);
        
        // Assume symmetric left-right distribution
        switch (idx) {
            case FRONT_LEFT:
            case FRONT_RIGHT:
                return front_load * 0.5;
            case REAR_LEFT:
            case REAR_RIGHT:
                return rear_load * 0.5;
        }
        return total_weight * 0.25;
    }
};

/**
 * @brief Tire model parameters (Pacejka-like)
 */
struct TireParams
{
    // Cornering stiffness (lateral force per slip angle)
    double C_alpha = 50000.0;  // [N/rad] Initial cornering stiffness
    
    // Longitudinal stiffness
    double C_kappa = 100000.0; // [N/-] Longitudinal slip stiffness
    
    // Peak friction coefficient
    double mu_peak = 0.9;      // [-] Peak friction coefficient
    double mu_slide = 0.7;     // [-] Sliding friction coefficient
    
    // Pacejka Magic Formula coefficients (simplified)
    // Fy = D * sin(C * atan(B * alpha - E * (B * alpha - atan(B * alpha))))
    double B = 10.0;   // Stiffness factor
    double C = 1.9;    // Shape factor
    double D = 1.0;    // Peak factor (= mu * Fz, computed dynamically)
    double E = 0.97;   // Curvature factor
    
    // Combined slip parameters
    double kappa_peak = 0.1;   // [-] Slip ratio at peak longitudinal force
    double alpha_peak = 0.087; // [rad] Slip angle at peak lateral force (~5 deg)
    
    // Relaxation length (for dynamics)
    double sigma_alpha = 0.3;  // [m] Lateral relaxation length
    double sigma_kappa = 0.1;  // [m] Longitudinal relaxation length
};

/**
 * @brief Velocity and acceleration limits
 */
struct VehicleLimits
{
    // Velocity limits
    double vx_max = 3.0;      // [m/s]
    double vy_max = 3.0;      // [m/s]
    double omega_max = 3.0;   // [rad/s]
    
    // Acceleration limits
    double ax_max = 5.0;      // [m/s²]
    double ay_max = 5.0;      // [m/s²]
    double alpha_max = 10.0;  // [rad/s²]
    
    // Wheel limits
    double wheel_vel_max = 10.0;  // [m/s]
    double steer_max = M_PI / 2;  // [rad]
    double steer_rate_max = 5.0;  // [rad/s]
};

/**
 * @brief Complete vehicle parameters
 */
struct VehicleParams
{
    VehicleGeometry geometry;
    VehicleMass mass;
    TireParams tire;
    VehicleLimits limits;

    // Force limits derived from vehicle properties
    double getMaxLongForce() const { return mass.mass * limits.ax_max; }
    double getMaxLatForce() const { return mass.mass * limits.ay_max; }
    double getMaxYawMoment() const { return mass.Iz * limits.alpha_max; }
};

// ============================================================================
// MPPI Parameters
// ============================================================================

/**
 * @brief MPPI sampling parameters
 */
struct MPPIParams
{
    int num_samples = 3000;        // K: Number of trajectory samples
    int prediction_horizon = 40;   // T: Prediction steps
    double step_dt = 0.033;        // [s] Time step
    double exploration_ratio = 0.1; // Ratio of pure exploration samples
    
    // MPPI algorithm
    double lambda = 100.0;         // Temperature parameter
    double alpha = 0.975;          // Control smoothing factor
    
    // Noise standard deviations (in force space)
    Eigen::Vector3d sigma_force = Eigen::Vector3d(200.0, 200.0, 50.0); // [N, N, Nm]
    
    // Alternative: noise in velocity space (for hybrid mode)
    Eigen::Vector3d sigma_vel = Eigen::Vector3d(0.5, 0.5, 0.8); // [m/s, m/s, rad/s]
    
    // Reference velocity
    double ref_velocity = 2.0;     // [m/s]
    
    // Sampling mode
    bool sample_in_force_space = true;  // true: force space, false: velocity space
};

/**
 * @brief Cost function weights
 */
struct CostWeights
{
    // Path tracking
    double distance_error = 60.0;     // Distance to reference path
    double angular_error = 30.0;      // Heading error
    double velocity_error = 10.0;     // Velocity tracking
    double terminal_state = 10.0;     // Terminal state cost
    
    // Safety
    double collision_penalty = 100.0; // Collision cost
    
    // Tire force related
    double force_utilization = 20.0;  // Penalize high friction utilization
    double friction_margin = 30.0;    // Reward maintaining friction margin
    double slip_angle = 15.0;         // Penalize large slip angles
    double slip_ratio = 10.0;         // Penalize large slip ratios
    
    // Curvature-aware speed
    double curvature_speed = 40.0;    // Speed regulation in curves
    double yaw_rate_tracking = 25.0;  // Yaw rate matching
    
    // Smoothness
    double force_change = 5.0;        // Force rate of change penalty
    double jerk = 2.0;                // Acceleration derivative penalty
    std::array<double, 8> wheel_cmd_change = {1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5};
};

/**
 * @brief Online parameter estimation settings
 */
struct EstimatorParams
{
    // Learning rates
    double lr_cornering = 0.01;    // Cornering stiffness learning rate
    double lr_friction = 0.005;    // Friction coefficient learning rate
    
    // Parameter bounds
    double C_alpha_min = 10000.0;  // [N/rad]
    double C_alpha_max = 100000.0; // [N/rad]
    double mu_min = 0.1;           // [-]
    double mu_max = 1.2;           // [-]
    
    // Excitation thresholds
    double min_speed = 0.3;        // [m/s] Minimum speed for estimation
    double min_lat_accel = 0.5;    // [m/s²] Minimum lateral acceleration
    double excitation_threshold = 0.1;  // Combined excitation threshold
    
    // Filtering
    double lpf_alpha = 0.1;        // Low-pass filter coefficient
    int convergence_window = 50;   // Samples for convergence check
    double convergence_threshold = 0.01; // Variance threshold
};

/**
 * @brief Force allocation settings
 */
struct AllocationParams
{
    // Optimization weights
    double w_force_min = 1.0;      // Minimize total force magnitude
    double w_friction_margin = 2.0; // Maximize friction margin
    double w_force_diff = 0.5;     // Minimize inter-wheel force difference
    
    // Constraints
    double friction_margin_target = 0.2;  // Target margin from friction limit
    int max_iterations = 10;       // Max optimization iterations
    double tolerance = 1e-3;       // Convergence tolerance
};

/**
 * @brief Savitzky-Golay filter parameters
 */
struct SGFilterParams
{
    bool enable = true;
    int half_window = 4;
    int poly_order = 3;
};

/**
 * @brief Complete controller configuration
 */
struct ControllerConfig
{
    VehicleParams vehicle;
    MPPIParams mppi;
    CostWeights weights;
    EstimatorParams estimator;
    AllocationParams allocation;
    SGFilterParams sg_filter;
    
    // Navigation tolerances
    double xy_goal_tolerance = 0.5;   // [m]
    double yaw_goal_tolerance = 0.5;  // [rad]
    
    // Control rate
    double control_rate = 50.0;       // [Hz]
};

// ============================================================================
// Collection Types
// ============================================================================

using State = FullState;
using Control = BodyForce;  // Primary control is in force space
using VelocityControl = BodyVelocity;

using StateSequence = std::vector<State>;
using ControlSequence = std::vector<Control>;
using VelocitySequence = std::vector<VelocityControl>;
using StateTrajectories = std::vector<StateSequence>;
using ControlTrajectories = std::vector<ControlSequence>;
using CostVector = std::vector<double>;

// ============================================================================
// Utility Structures
// ============================================================================

/**
 * @brief Estimation statistics for monitoring
 */
struct EstimatorStats
{
    double C_alpha_est = 50000.0;  // Estimated cornering stiffness
    double mu_est = 0.9;           // Estimated friction coefficient
    double estimation_error = 0.0; // Current estimation error
    bool is_converged = false;     // Convergence flag
    int num_samples = 0;           // Number of update samples
    double residual_variance = 1.0; // Error variance
};

/**
 * @brief Force allocation result
 */
struct AllocationResult
{
    WheelForces wheel_forces;      // Allocated forces per wheel
    BodyForce achieved_force;      // Actually achievable body force
    double total_utilization = 0.0; // Average friction utilization
    double min_margin = 1.0;       // Minimum friction margin
    bool is_saturated = false;     // Any wheel saturated?
    int iterations = 0;            // Solver iterations
};

/**
 * @brief Per-step prediction data (for analysis)
 */
struct PredictionStep
{
    State state;
    Control control;
    WheelForces wheel_forces;
    std::array<double, NUM_WHEELS> slip_angles;
    std::array<double, NUM_WHEELS> friction_utilization;
    double cost = 0.0;
};

} // namespace mppi_tf
