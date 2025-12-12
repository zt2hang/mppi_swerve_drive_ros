/**
 * @file mppi_tf_core.cpp
 * @brief Implementation of MPPI-TireForce core controller
 * 
 * @author ZZT
 * @date 2024
 */

#include "mppi_tf/mppi_tf_core.hpp"
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace mppi_tf
{

MPPITFCore::MPPITFCore(const ControllerConfig& config)
    : config_(config)
    , dynamics_(config.vehicle, DynamicsMode::FORCE_BASED)
    , tire_model_(config.vehicle.tire)
    , force_allocator_(config.vehicle.geometry, config.vehicle.mass, config.allocation)
    , estimator_(config.estimator, config.vehicle.geometry, config.vehicle.mass)
    , cost_function_(config.weights, config.vehicle)
    , rng_(623)  // Fixed seed for reproducibility
{
    initializeMPPI();
    if (config_.sg_filter.enable) {
        initSGFilter();
    }
}

void MPPITFCore::initializeMPPI()
{
    K_ = config_.mppi.num_samples;
    T_ = config_.mppi.prediction_horizon;

    // Allocate storage
    state_samples_.resize(K_, StateSequence(T_));
    force_samples_.resize(K_, ControlSequence(T_));
    vel_samples_.resize(K_, VelocitySequence(T_));
    noise_force_.resize(K_, ControlSequence(T_));
    noise_vel_.resize(K_, VelocitySequence(T_));
    costs_.resize(K_, 0.0);
    cost_ranks_.resize(K_);
    weights_.resize(K_, 0.0);

    // Initialize sequences
    optimal_force_seq_.resize(T_);
    optimal_velocity_seq_.resize(T_);
    optimal_trajectory_.resize(T_);
    velocity_history_.resize(config_.sg_filter.half_window);

    for (int t = 0; t < T_; ++t) {
        optimal_force_seq_[t].setZero();
        optimal_velocity_seq_[t].setZero();
    }

    last_command_.setZero();
    last_force_.setZero();

    std::iota(cost_ranks_.begin(), cost_ranks_.end(), 0);
}

void MPPITFCore::initSGFilter()
{
    // Savitzky-Golay filter coefficients
    int hw = config_.sg_filter.half_window;
    int order = config_.sg_filter.poly_order;
    int window_size = 2 * hw + 1;

    // Build Vandermonde matrix
    Eigen::MatrixXd V(window_size, order + 1);
    for (int i = 0; i < window_size; ++i) {
        int x = i - hw;
        for (int j = 0; j <= order; ++j) {
            V(i, j) = std::pow(x, j);
        }
    }

    // Compute filter coefficients via least squares
    // coeffs = (V^T V)^{-1} V^T
    Eigen::MatrixXd VtV = V.transpose() * V;
    sg_coeffs_ = VtV.ldlt().solve(V.transpose());
}

void MPPITFCore::setConfig(const ControllerConfig& config)
{
    config_ = config;
    dynamics_ = DynamicsModel(config.vehicle, DynamicsMode::FORCE_BASED);
    tire_model_ = TireModel(config.vehicle.tire);
    force_allocator_ = ForceAllocator(config.vehicle.geometry, config.vehicle.mass, config.allocation);
    cost_function_.setWeights(config.weights);
    
    initializeMPPI();
    if (config_.sg_filter.enable) {
        initSGFilter();
    }
}

void MPPITFCore::setFeedbackGains(double k_lat, double k_head, double k_integral)
{
    k_lateral_ = k_lat;
    k_heading_ = k_head;
    k_integral_ = k_integral;
}

// ============================================================================
// Main Solve Interface
// ============================================================================

BodyVelocity MPPITFCore::solve(
    const FullState& current_state,
    const grid_map::GridMap& collision_map,
    const grid_map::GridMap& distance_map,
    const grid_map::GridMap& yaw_map,
    const FullState& goal)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    // Check goal reached
    if (checkGoalReached(current_state, goal)) {
        goal_reached_ = true;
        last_command_.setZero();
        return last_command_;
    }
    goal_reached_ = false;

    // Update dynamics with current tire parameters
    dynamics_.setTireParams(
        estimator_.getCorneringStiffness(),
        estimator_.getFrictionCoeff()
    );
    cost_function_.setFrictionCoeff(estimator_.getFrictionCoeff());
    force_allocator_.setFrictionCoeff(estimator_.getFrictionCoeff());

    // Generate noise samples
    generateNoiseSamples();

    // Rollout trajectories based on sampling mode
    if (sampling_mode_ == SamplingMode::FORCE_SPACE) {
        rolloutTrajectoriesForce(current_state, collision_map, distance_map, yaw_map, goal);
    } else {
        rolloutTrajectoriesVelocity(current_state, collision_map, distance_map, yaw_map, goal);
    }

    // Compute weights and update optimal sequence
    computeWeights();
    updateOptimalSequence();

    // Get command
    BodyVelocity cmd;
    if (sampling_mode_ == SamplingMode::FORCE_SPACE) {
        cmd = forceToVelocity(optimal_force_seq_[0], current_state, config_.mppi.step_dt);
    } else {
        cmd = optimal_velocity_seq_[0];
    }

    // Apply smoothing filter
    if (config_.sg_filter.enable) {
        velocity_history_.push_back(cmd);
        if (velocity_history_.size() > static_cast<size_t>(2 * config_.sg_filter.half_window + 1)) {
            velocity_history_.erase(velocity_history_.begin());
        }
        cmd = applySGFilter(velocity_history_);
    }

    // Rate limiter: prevent sudden changes in control
    double max_dvx = config_.vehicle.limits.ax_max * config_.mppi.step_dt;
    double max_dvy = config_.vehicle.limits.ay_max * config_.mppi.step_dt;
    double max_domega = 3.0 * config_.mppi.step_dt;  // Max angular acceleration

    cmd.vx = last_command_.vx + std::clamp(cmd.vx - last_command_.vx, -max_dvx, max_dvx);
    cmd.vy = last_command_.vy + std::clamp(cmd.vy - last_command_.vy, -max_dvy, max_dvy);
    cmd.omega = last_command_.omega + std::clamp(cmd.omega - last_command_.omega, -max_domega, max_domega);

    // Exponential smoothing with previous command
    double alpha_smooth = config_.mppi.alpha;
    cmd.vx = alpha_smooth * cmd.vx + (1.0 - alpha_smooth) * last_command_.vx;
    cmd.vy = alpha_smooth * cmd.vy + (1.0 - alpha_smooth) * last_command_.vy;
    cmd.omega = alpha_smooth * cmd.omega + (1.0 - alpha_smooth) * last_command_.omega;

    // Clamp command
    cmd.clamp(config_.vehicle.limits.vx_max, 
              config_.vehicle.limits.vy_max, 
              config_.vehicle.limits.omega_max);

    last_command_ = cmd;

    // Record timing
    auto end_time = std::chrono::high_resolution_clock::now();
    calc_time_ms_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return cmd;
}

BodyVelocity MPPITFCore::solveWithFeedback(
    const FullState& current_state,
    const grid_map::GridMap& collision_map,
    const grid_map::GridMap& distance_map,
    const grid_map::GridMap& yaw_map,
    const FullState& goal,
    double lateral_error,
    double heading_error,
    double path_curvature,
    double dt)
{
    // Get base command from MPPI
    BodyVelocity cmd = solve(current_state, collision_map, distance_map, yaw_map, goal);

    // Apply feedback compensation
    cmd = applyFeedback(cmd, lateral_error, heading_error, path_curvature, dt);

    return cmd;
}

// ============================================================================
// Noise Generation
// ============================================================================

void MPPITFCore::generateNoiseSamples()
{
    const auto& sigma_f = config_.mppi.sigma_force;
    const auto& sigma_v = config_.mppi.sigma_vel;

    std::normal_distribution<double> dist_fx(0.0, sigma_f(0));
    std::normal_distribution<double> dist_fy(0.0, sigma_f(1));
    std::normal_distribution<double> dist_mz(0.0, sigma_f(2));
    
    std::normal_distribution<double> dist_vx(0.0, sigma_v(0));
    std::normal_distribution<double> dist_vy(0.0, sigma_v(1));
    std::normal_distribution<double> dist_omega(0.0, sigma_v(2));

    #pragma omp parallel for num_threads(omp_get_max_threads())
    for (int k = 0; k < K_; ++k) {
        // Thread-local RNG
        std::mt19937 local_rng(rng_() + k);

        for (int t = 0; t < T_; ++t) {
            noise_force_[k][t].Fx = dist_fx(local_rng);
            noise_force_[k][t].Fy = dist_fy(local_rng);
            noise_force_[k][t].Mz = dist_mz(local_rng);

            noise_vel_[k][t].vx = dist_vx(local_rng);
            noise_vel_[k][t].vy = dist_vy(local_rng);
            noise_vel_[k][t].omega = dist_omega(local_rng);
        }
    }
}

// ============================================================================
// Trajectory Rollout - Force Space
// ============================================================================

void MPPITFCore::rolloutTrajectoriesForce(
    const FullState& initial,
    const grid_map::GridMap& collision_map,
    const grid_map::GridMap& distance_map,
    const grid_map::GridMap& yaw_map,
    const FullState& goal)
{
    double dt = config_.mppi.step_dt;
    double exploration_ratio = config_.mppi.exploration_ratio;
    int exploration_start = static_cast<int>((1.0 - exploration_ratio) * K_);

    auto force_limits = dynamics_.getForceLimits();

    #pragma omp parallel for num_threads(omp_get_max_threads())
    for (int k = 0; k < K_; ++k) {
        FullState x = initial;
        double cost = 0.0;
        BodyForce prev_force = last_force_;

        for (int t = 0; t < T_; ++t) {
            // Sample force
            BodyForce force;
            if (k < exploration_start) {
                // Exploitation: perturb around previous optimal
                force.Fx = optimal_force_seq_[t].Fx + noise_force_[k][t].Fx;
                force.Fy = optimal_force_seq_[t].Fy + noise_force_[k][t].Fy;
                force.Mz = optimal_force_seq_[t].Mz + noise_force_[k][t].Mz;
            } else {
                // Exploration: pure noise
                force = noise_force_[k][t];
            }

            // Clamp force
            force.clamp(force_limits.Fx, force_limits.Fy, force_limits.Mz);
            force_samples_[k][t] = force;

            // Allocate force to wheels
            auto alloc_result = force_allocator_.allocate(force);

            // Propagate dynamics
            x = dynamics_.stepForce(x, alloc_result.achieved_force, dt);
            state_samples_[k][t] = x;

            // Accumulate cost
            cost += cost_function_.stageCostForce(
                x, force, prev_force, alloc_result.wheel_forces,
                collision_map, distance_map, yaw_map, goal
            );

            // Information-theoretic term
            const auto& sigma = config_.mppi.sigma_force;
            Eigen::Vector3d sigma_inv(1.0/sigma(0), 1.0/sigma(1), 1.0/sigma(2));
            Eigen::Vector3d f_opt = optimal_force_seq_[t].toEigen();
            Eigen::Vector3d f_curr = force.toEigen();
            cost += config_.mppi.lambda * (1.0 - config_.mppi.alpha) *
                    f_opt.cwiseProduct(sigma_inv).dot(f_curr);

            prev_force = force;
        }

        // Terminal cost
        cost += cost_function_.terminalCost(x, goal);
        costs_[k] = cost;
    }
}

// ============================================================================
// Trajectory Rollout - Velocity Space
// ============================================================================

void MPPITFCore::rolloutTrajectoriesVelocity(
    const FullState& initial,
    const grid_map::GridMap& collision_map,
    const grid_map::GridMap& distance_map,
    const grid_map::GridMap& yaw_map,
    const FullState& goal)
{
    double dt = config_.mppi.step_dt;
    double exploration_ratio = config_.mppi.exploration_ratio;
    int exploration_start = static_cast<int>((1.0 - exploration_ratio) * K_);

    // Slip factor from estimator (simplified slip model)
    double slip_factor = estimator_.getCorneringStiffness() > 30000 ? 0.1 : 0.2;

    #pragma omp parallel for num_threads(omp_get_max_threads())
    for (int k = 0; k < K_; ++k) {
        FullState x = initial;
        double cost = 0.0;
        BodyVelocity prev_vel = last_command_;

        for (int t = 0; t < T_; ++t) {
            // Sample velocity
            BodyVelocity vel;
            if (k < exploration_start) {
                // Exploitation
                vel.vx = optimal_velocity_seq_[t].vx + noise_vel_[k][t].vx;
                vel.vy = optimal_velocity_seq_[t].vy + noise_vel_[k][t].vy;
                vel.omega = optimal_velocity_seq_[t].omega + noise_vel_[k][t].omega;
            } else {
                // Exploration
                vel = noise_vel_[k][t];
            }

            // Clamp
            vel.clamp(config_.vehicle.limits.vx_max,
                     config_.vehicle.limits.vy_max,
                     config_.vehicle.limits.omega_max);
            vel_samples_[k][t] = vel;

            // Propagate dynamics with slip
            x = dynamics_.stepWithSlip(x, vel, dt, slip_factor);
            state_samples_[k][t] = x;

            // Accumulate cost
            cost += cost_function_.stageCostVelocity(
                x, vel, prev_vel,
                collision_map, distance_map, yaw_map, goal, slip_factor
            );

            // Information-theoretic term
            const auto& sigma = config_.mppi.sigma_vel;
            Eigen::Vector3d sigma_inv(1.0/sigma(0), 1.0/sigma(1), 1.0/sigma(2));
            Eigen::Vector3d v_opt = optimal_velocity_seq_[t].toEigen();
            Eigen::Vector3d v_curr = vel.toEigen();
            cost += config_.mppi.lambda * (1.0 - config_.mppi.alpha) *
                    v_opt.cwiseProduct(sigma_inv).dot(v_curr);

            prev_vel = vel;
        }

        // Terminal cost
        cost += cost_function_.terminalCost(x, goal);
        costs_[k] = cost;
    }
}

// ============================================================================
// Weight Computation
// ============================================================================

void MPPITFCore::computeWeights()
{
    // Find minimum cost
    double min_cost = *std::min_element(costs_.begin(), costs_.end());

    // Compute softmax weights
    double eta = 0.0;
    for (int k = 0; k < K_; ++k) {
        weights_[k] = std::exp(-1.0 / config_.mppi.lambda * (costs_[k] - min_cost));
        eta += weights_[k];
    }

    // Normalize
    for (int k = 0; k < K_; ++k) {
        weights_[k] /= (eta + EPSILON);
    }

    // Update state cost (for monitoring)
    state_cost_ = min_cost;

    // Sort by cost for elite sampling
    std::iota(cost_ranks_.begin(), cost_ranks_.end(), 0);
    std::partial_sort(cost_ranks_.begin(), cost_ranks_.begin() + std::min(K_, 100),
                      cost_ranks_.end(),
                      [this](int a, int b) { return costs_[a] < costs_[b]; });
}

// ============================================================================
// Optimal Sequence Update
// ============================================================================

void MPPITFCore::updateOptimalSequence()
{
    // Weighted average of control sequences
    if (sampling_mode_ == SamplingMode::FORCE_SPACE) {
        for (int t = 0; t < T_; ++t) {
            optimal_force_seq_[t].setZero();
            for (int k = 0; k < K_; ++k) {
                optimal_force_seq_[t].Fx += weights_[k] * force_samples_[k][t].Fx;
                optimal_force_seq_[t].Fy += weights_[k] * force_samples_[k][t].Fy;
                optimal_force_seq_[t].Mz += weights_[k] * force_samples_[k][t].Mz;
            }
        }
        last_force_ = optimal_force_seq_[0];
    } else {
        for (int t = 0; t < T_; ++t) {
            optimal_velocity_seq_[t].setZero();
            for (int k = 0; k < K_; ++k) {
                optimal_velocity_seq_[t].vx += weights_[k] * vel_samples_[k][t].vx;
                optimal_velocity_seq_[t].vy += weights_[k] * vel_samples_[k][t].vy;
                optimal_velocity_seq_[t].omega += weights_[k] * vel_samples_[k][t].omega;
            }
        }
    }

    // Shift sequence (warm start for next iteration)
    if (sampling_mode_ == SamplingMode::FORCE_SPACE) {
        std::rotate(optimal_force_seq_.begin(), optimal_force_seq_.begin() + 1, optimal_force_seq_.end());
        optimal_force_seq_.back() = optimal_force_seq_[T_-2];  // Repeat last
    } else {
        std::rotate(optimal_velocity_seq_.begin(), optimal_velocity_seq_.begin() + 1, optimal_velocity_seq_.end());
        optimal_velocity_seq_.back() = optimal_velocity_seq_[T_-2];
    }

    // Update optimal trajectory (from best sample)
    int best_idx = cost_ranks_[0];
    optimal_trajectory_ = state_samples_[best_idx];
}

// ============================================================================
// Utility Functions
// ============================================================================

BodyVelocity MPPITFCore::forceToVelocity(
    const BodyForce& force, 
    const FullState& state, 
    double dt)
{
    // Compute acceleration from force
    double ax = force.Fx / config_.vehicle.mass.mass;
    double ay = force.Fy / config_.vehicle.mass.mass;
    double alpha = force.Mz / config_.vehicle.mass.Iz;

    // Integrate to get velocity change
    BodyVelocity vel;
    vel.vx = state.vx + ax * dt;
    vel.vy = state.vy + ay * dt;
    vel.omega = state.omega + alpha * dt;

    return vel;
}

BodyVelocity MPPITFCore::applySGFilter(const VelocitySequence& seq)
{
    if (seq.size() < static_cast<size_t>(2 * config_.sg_filter.half_window + 1)) {
        return seq.back();
    }

    // Apply filter (0th derivative = smoothing)
    int hw = config_.sg_filter.half_window;
    Eigen::VectorXd vx_vec(2 * hw + 1);
    Eigen::VectorXd vy_vec(2 * hw + 1);
    Eigen::VectorXd omega_vec(2 * hw + 1);

    for (int i = 0; i < 2 * hw + 1; ++i) {
        int idx = seq.size() - (2 * hw + 1) + i;
        vx_vec(i) = seq[idx].vx;
        vy_vec(i) = seq[idx].vy;
        omega_vec(i) = seq[idx].omega;
    }

    // Smoothed values (0th row of coefficients)
    BodyVelocity filtered;
    filtered.vx = sg_coeffs_.row(0).dot(vx_vec);
    filtered.vy = sg_coeffs_.row(0).dot(vy_vec);
    filtered.omega = sg_coeffs_.row(0).dot(omega_vec);

    return filtered;
}

BodyVelocity MPPITFCore::applyFeedback(
    const BodyVelocity& planned,
    double lateral_error,
    double heading_error,
    double path_curvature,
    double dt)
{
    // Dead zone: ignore very small errors to prevent oscillation
    const double lat_deadzone = 0.02;   // 2cm
    const double head_deadzone = 0.02;  // ~1 degree

    double lat_err_filtered = (std::abs(lateral_error) < lat_deadzone) ? 0.0 : lateral_error;
    double head_err_filtered = (std::abs(heading_error) < head_deadzone) ? 0.0 : heading_error;

    // Update integral term with anti-windup
    if (std::abs(lat_err_filtered) > 0.01) {
        error_integral_ += lat_err_filtered * dt;
        error_integral_ = std::clamp(error_integral_, -0.5, 0.5);
    } else {
        // Decay integral when error is small
        error_integral_ *= 0.95;
    }

    // Feedback compensation - proportional to speed for stability
    double speed = planned.speed();
    double speed_factor = std::clamp(speed / 1.0, 0.0, 1.0);  // Ramp up feedback with speed

    double dvy_fb = -k_lateral_ * lat_err_filtered * speed_factor
                   - k_heading_ * head_err_filtered * speed
                   - k_integral_ * error_integral_;

    // Limit feedback magnitude to prevent instability
    double max_feedback = 0.3;  // Max 0.3 m/s correction
    dvy_fb = std::clamp(dvy_fb, -max_feedback, max_feedback);

    // Apply compensation
    BodyVelocity compensated = planned;
    compensated.vy += dvy_fb;

    return compensated;
}

bool MPPITFCore::checkGoalReached(const FullState& state, const FullState& goal)
{
    double pos_error = std::hypot(state.x - goal.x, state.y - goal.y);
    double yaw_error = std::abs(std::remainder(state.yaw - goal.yaw, 2.0 * M_PI));

    return (pos_error < config_.xy_goal_tolerance) &&
           (yaw_error < config_.yaw_goal_tolerance);
}

void MPPITFCore::updateEstimator(
    const BodyVelocity& cmd_vel,
    const BodyVelocity& actual_vel,
    double dt)
{
    estimator_.updateFromVelocity(cmd_vel, actual_vel, dt);

    // Update tire model with new estimates
    tire_model_.setCorneringStiffness(estimator_.getCorneringStiffness());
    tire_model_.setFrictionCoeff(estimator_.getFrictionCoeff());
}

EstimatorStats MPPITFCore::getEstimatorStats() const
{
    return estimator_.getStatistics();
}

StateTrajectories MPPITFCore::getEliteTrajectories(int n) const
{
    StateTrajectories elite;
    int count = std::min(n, K_);
    
    for (int i = 0; i < count; ++i) {
        elite.push_back(state_samples_[cost_ranks_[i]]);
    }
    
    return elite;
}

VehicleCommand8D MPPITFCore::getWheelCommands() const
{
    return dynamics_.velocityToWheelCommands(last_command_);
}

} // namespace mppi_tf
