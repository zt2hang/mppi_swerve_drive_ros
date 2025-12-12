/**
 * @file mppi_hc_core.cpp
 * @brief Implementation of MPPI-HC core controller
 */

#include "mppi_hc/mppi_hc_core.hpp"
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace mppi_hc
{

MPPIHCCore::MPPIHCCore(const ControllerConfig& config)
    : config_(config)
    , dynamics_(config.vehicle)
    , cost_function_(config.weights, config.slip, config.vehicle)
    , slip_estimator_(config.slip)
    , slip_compensator_(config.slip)
    , rng_(623)  // Fixed seed for reproducibility
{
    initializeMPPI();
    if (config_.use_sg_filter) {
        initSGFilter();
    }
}

void MPPIHCCore::initializeMPPI()
{
    K_ = config_.mppi.num_samples;
    T_ = config_.mppi.prediction_horizon;

    // Allocate storage
    state_samples_.resize(K_, StateSequence(T_));
    control_samples_.resize(K_, ControlSequence(T_));
    noise_samples_.resize(K_, ControlSequence(T_));
    costs_.resize(K_, 0.0);
    cost_ranks_.resize(K_);
    weights_.resize(K_, 0.0);

    optimal_control_seq_.resize(T_);
    optimal_trajectory_.resize(T_);
    prior_control_seq_.resize(T_);
    control_history_.resize(config_.sg_half_window);

    // Initialize control sequences to zero
    for (int t = 0; t < T_; ++t) {
        optimal_control_seq_[t].setZero();
        prior_control_seq_[t].setZero();
    }
    last_command_.setZero();

    std::iota(cost_ranks_.begin(), cost_ranks_.end(), 0);
}

void MPPIHCCore::clearControlPrior()
{
    prior_enabled_ = false;
    prior_weight_ = 0.0;
    prior_apply_to_exploration_ = true;
    for (auto& u : prior_control_seq_) {
        u.setZero();
    }
}

void MPPIHCCore::setControlPrior(const ControlSequence& prior_sequence,
                                 double regularization_weight,
                                 bool apply_to_exploration)
{
    if (prior_sequence.size() != static_cast<std::size_t>(T_)) {
        // Size mismatch: ignore to avoid out-of-bounds.
        return;
    }
    prior_control_seq_ = prior_sequence;
    prior_enabled_ = true;
    prior_weight_ = std::max(0.0, regularization_weight);
    prior_apply_to_exploration_ = apply_to_exploration;
}

void MPPIHCCore::generateNoiseSamples()
{
    const auto& sigma = config_.mppi.sigma;
    
    std::normal_distribution<double> dist_vx(0.0, sigma(0));
    std::normal_distribution<double> dist_vy(0.0, sigma(1));
    std::normal_distribution<double> dist_omega(0.0, sigma(2));

    #pragma omp parallel for num_threads(omp_get_max_threads())
    for (int k = 0; k < K_; ++k) {
        // Use thread-local RNG for thread safety
        std::mt19937 local_rng(rng_() + k);
        
        for (int t = 0; t < T_; ++t) {
            noise_samples_[k][t].vx = dist_vx(local_rng);
            noise_samples_[k][t].vy = dist_vy(local_rng);
            noise_samples_[k][t].omega = dist_omega(local_rng);
        }
    }
}

void MPPIHCCore::rolloutTrajectories(
    const State& initial_state,
    const grid_map::GridMap& collision_map,
    const grid_map::GridMap& distance_error_map,
    const grid_map::GridMap& ref_yaw_map,
    const State& goal
)
{
    double slip_factor = slip_estimator_.getSlipFactor();
    double dt = config_.mppi.step_dt;
    double exploration_ratio = config_.mppi.exploration_ratio;
    int exploration_start = static_cast<int>((1.0 - exploration_ratio) * K_);

    #pragma omp parallel for num_threads(omp_get_max_threads())
    for (int k = 0; k < K_; ++k) {
        State x = initial_state;
        double cost = 0.0;

        for (int t = 0; t < T_; ++t) {
            Control u_center = optimal_control_seq_[t];
            if (prior_enabled_ && prior_control_seq_.size() == static_cast<std::size_t>(T_)) {
                u_center.vx += prior_control_seq_[t].vx;
                u_center.vy += prior_control_seq_[t].vy;
                u_center.omega += prior_control_seq_[t].omega;
            }

            // Sample control
            Control u;
            if (k < exploration_start) {
                // Exploitation: perturb around previous optimal
                u.vx = u_center.vx + noise_samples_[k][t].vx;
                u.vy = u_center.vy + noise_samples_[k][t].vy;
                u.omega = u_center.omega + noise_samples_[k][t].omega;
            } else {
                // Exploration: pure noise
                if (prior_enabled_ && prior_apply_to_exploration_ &&
                    prior_control_seq_.size() == static_cast<std::size_t>(T_)) {
                    u.vx = prior_control_seq_[t].vx + noise_samples_[k][t].vx;
                    u.vy = prior_control_seq_[t].vy + noise_samples_[k][t].vy;
                    u.omega = prior_control_seq_[t].omega + noise_samples_[k][t].omega;
                } else {
                    u = noise_samples_[k][t];
                }
            }
            u.clamp(config_.vehicle.vx_max, config_.vehicle.vy_max, config_.vehicle.omega_max);
            
            control_samples_[k][t] = u;

            // Propagate dynamics
            x = dynamics_.step(x, u, dt, slip_factor);
            state_samples_[k][t] = x;

            // Accumulate stage cost
            Control prev_u = (t == 0) ? last_command_ : control_samples_[k][t-1];
            cost += cost_function_.stageCost(x, u, prev_u, collision_map, 
                                             distance_error_map, ref_yaw_map, goal, slip_factor);

            // Optional: keep samples close to the prior-centered mean
            if (prior_enabled_ && prior_weight_ > 0.0) {
                Eigen::Vector3d du = u.toEigen() - u_center.toEigen();
                cost += prior_weight_ * du.squaredNorm();
            }

            // Information-theoretic cost term
            const auto& sigma = config_.mppi.sigma;
            Eigen::Vector3d sigma_inv(1.0/sigma(0), 1.0/sigma(1), 1.0/sigma(2));
                Eigen::Vector3d u_opt = u_center.toEigen();
            Eigen::Vector3d u_curr = u.toEigen();
            cost += config_.mppi.lambda * (1.0 - config_.mppi.alpha) * 
                    u_opt.cwiseProduct(sigma_inv).dot(u_curr);
        }

        // Terminal cost
        cost += cost_function_.terminalCost(x, goal);
        costs_[k] = cost;
    }
}

void MPPIHCCore::computeWeights()
{
    // Find minimum cost
    double min_cost = *std::min_element(costs_.begin(), costs_.end());

    // Compute unnormalized weights (softmax)
    double eta = 0.0;
    for (int k = 0; k < K_; ++k) {
        weights_[k] = std::exp(-1.0 / config_.mppi.lambda * (costs_[k] - min_cost));
        eta += weights_[k];
    }

    // Normalize
    for (int k = 0; k < K_; ++k) {
        weights_[k] /= eta;
    }

    // Update cost rankings
    std::iota(cost_ranks_.begin(), cost_ranks_.end(), 0);
    std::sort(cost_ranks_.begin(), cost_ranks_.end(),
              [this](int a, int b) { return costs_[a] < costs_[b]; });
}

void MPPIHCCore::updateOptimalSequence()
{
    // Weighted average of noise
    for (int t = 0; t < T_; ++t) {
        Eigen::Vector3d weighted_noise = Eigen::Vector3d::Zero();
        for (int k = 0; k < K_; ++k) {
            weighted_noise += weights_[k] * noise_samples_[k][t].toEigen();
        }
        
        Eigen::Vector3d u_new = optimal_control_seq_[t].toEigen() + weighted_noise;
        optimal_control_seq_[t].fromEigen(u_new);
        optimal_control_seq_[t].clamp(config_.vehicle.vx_max, 
                                       config_.vehicle.vy_max, 
                                       config_.vehicle.omega_max);
    }
}

BodyVelocity MPPIHCCore::solve(
    const State& current_state,
    const grid_map::GridMap& collision_map,
    const grid_map::GridMap& distance_error_map,
    const grid_map::GridMap& ref_yaw_map,
    const State& goal
)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    // Check if goal reached
    double dist_to_goal = std::sqrt(std::pow(goal.x - current_state.x, 2) + 
                                    std::pow(goal.y - current_state.y, 2));
    double yaw_to_goal = std::abs(std::remainder(current_state.yaw - goal.yaw, 2.0 * M_PI));
    
    if (dist_to_goal < config_.xy_goal_tolerance && 
        yaw_to_goal < config_.yaw_goal_tolerance) {
        goal_reached_ = true;
        BodyVelocity stop_cmd;
        stop_cmd.setZero();
        return stop_cmd;
    }
    goal_reached_ = false;

    // MPPI optimization
    generateNoiseSamples();
    rolloutTrajectories(current_state, collision_map, distance_error_map, ref_yaw_map, goal);
    computeWeights();
    updateOptimalSequence();

    // Build total control sequence (optimal + prior)
    ControlSequence total_seq = optimal_control_seq_;
    if (prior_enabled_ && prior_control_seq_.size() == static_cast<std::size_t>(T_)) {
        for (int t = 0; t < T_; ++t) {
            total_seq[t].vx += prior_control_seq_[t].vx;
            total_seq[t].vy += prior_control_seq_[t].vy;
            total_seq[t].omega += prior_control_seq_[t].omega;
            total_seq[t].clamp(config_.vehicle.vx_max, config_.vehicle.vy_max, config_.vehicle.omega_max);
        }
    }

    // Apply Savitzky-Golay filter if enabled
    Control u_optimal = total_seq[0];
    if (config_.use_sg_filter && sg_initialized_) {
        u_optimal = applyFilter(total_seq);
    }

    // =========================================================================
    // Layer 2: Feedforward Slip Compensation
    // =========================================================================
    double slip_factor = slip_estimator_.getSlipFactor();
    BodyVelocity compensated_cmd = slip_compensator_.compensate(u_optimal, slip_factor);

    // Clamp final command
    compensated_cmd.clamp(config_.vehicle.vx_max, config_.vehicle.vy_max, config_.vehicle.omega_max);

    // Update last command
    last_command_ = compensated_cmd;

    // Compute optimal trajectory for visualization
    State x = current_state;
    state_cost_ = 0.0;
    for (int t = 0; t < T_; ++t) {
        x = dynamics_.step(x, total_seq[t], config_.mppi.step_dt, slip_factor);
        optimal_trajectory_[t] = x;
        
        Control prev_u = (t == 0) ? last_command_ : total_seq[t-1];
        state_cost_ += cost_function_.stageCost(x, total_seq[t], prev_u,
                                                collision_map, distance_error_map, 
                                                ref_yaw_map, goal, slip_factor);
    }
    state_cost_ += cost_function_.terminalCost(x, goal);

    // Record timing
    auto end_time = std::chrono::high_resolution_clock::now();
    calc_time_ms_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return compensated_cmd;
}

BodyVelocity MPPIHCCore::solveWithFeedback(
    const State& current_state,
    const grid_map::GridMap& collision_map,
    const grid_map::GridMap& distance_error_map,
    const grid_map::GridMap& ref_yaw_map,
    const State& goal,
    double lateral_error,
    double heading_error,
    double path_curvature,
    double dt
)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    // Check if goal reached
    double dist_to_goal = std::sqrt(std::pow(goal.x - current_state.x, 2) + 
                                    std::pow(goal.y - current_state.y, 2));
    double yaw_to_goal = std::abs(std::remainder(current_state.yaw - goal.yaw, 2.0 * M_PI));
    
    if (dist_to_goal < config_.xy_goal_tolerance && 
        yaw_to_goal < config_.yaw_goal_tolerance) {
        goal_reached_ = true;
        slip_compensator_.resetIntegrator();
        BodyVelocity stop_cmd;
        stop_cmd.setZero();
        return stop_cmd;
    }
    goal_reached_ = false;
    
    // Near goal: aggressively decay integrator to prevent wandering
    const double goal_proximity_threshold = 1.0;  // [m]
    if (dist_to_goal < goal_proximity_threshold) {
        double decay_factor = 0.9 * (dist_to_goal / goal_proximity_threshold);  // 0 at goal, 0.9 at threshold
        slip_compensator_.decayIntegrator(decay_factor);
    }

    // MPPI optimization (Layer 1)
    generateNoiseSamples();
    rolloutTrajectories(current_state, collision_map, distance_error_map, ref_yaw_map, goal);
    computeWeights();
    updateOptimalSequence();

    // Build total control sequence (optimal + prior)
    ControlSequence total_seq = optimal_control_seq_;
    if (prior_enabled_ && prior_control_seq_.size() == static_cast<std::size_t>(T_)) {
        for (int t = 0; t < T_; ++t) {
            total_seq[t].vx += prior_control_seq_[t].vx;
            total_seq[t].vy += prior_control_seq_[t].vy;
            total_seq[t].omega += prior_control_seq_[t].omega;
            total_seq[t].clamp(config_.vehicle.vx_max, config_.vehicle.vy_max, config_.vehicle.omega_max);
        }
    }

    // Apply Savitzky-Golay filter if enabled
    Control u_optimal = total_seq[0];
    if (config_.use_sg_filter && sg_initialized_) {
        u_optimal = applyFilter(total_seq);
    }

    // =========================================================================
    // Layer 2 + 3: CLOSED-LOOP Slip Compensation (FF + FB)
    // =========================================================================
    double slip_factor = slip_estimator_.getSlipFactor();
    
    // Update error integrator
    slip_compensator_.updateError(lateral_error, dt);
    
    // Apply closed-loop compensation
    BodyVelocity compensated_cmd = slip_compensator_.compensateClosedLoop(
        u_optimal, slip_factor, lateral_error, heading_error, path_curvature);

    // Clamp final command
    compensated_cmd.clamp(config_.vehicle.vx_max, config_.vehicle.vy_max, config_.vehicle.omega_max);

    // Update last command
    last_command_ = compensated_cmd;

    // Compute optimal trajectory for visualization
    State x = current_state;
    state_cost_ = 0.0;
    for (int t = 0; t < T_; ++t) {
        x = dynamics_.step(x, total_seq[t], config_.mppi.step_dt, slip_factor);
        optimal_trajectory_[t] = x;
        
        Control prev_u = (t == 0) ? last_command_ : total_seq[t-1];
        state_cost_ += cost_function_.stageCost(x, total_seq[t], prev_u,
                                                collision_map, distance_error_map, 
                                                ref_yaw_map, goal, slip_factor);
    }
    state_cost_ += cost_function_.terminalCost(x, goal);

    // Record timing
    auto end_time = std::chrono::high_resolution_clock::now();
    calc_time_ms_ = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return compensated_cmd;
}

void MPPIHCCore::updateEstimator(double actual_vx, double actual_vy, double actual_omega)
{
    BodyVelocity actual_velocity;
    actual_velocity.vx = actual_vx;
    actual_velocity.vy = actual_vy;
    actual_velocity.omega = actual_omega;
    slip_estimator_.update(last_command_, actual_velocity, config_.mppi.step_dt);
}

StateTrajectories MPPIHCCore::getEliteTrajectories(int n) const
{
    StateTrajectories elite;
    n = std::min(n, K_);
    for (int i = 0; i < n; ++i) {
        elite.push_back(state_samples_[cost_ranks_[i]]);
    }
    return elite;
}

VehicleCommand8D MPPIHCCore::getWheelCommands() const
{
    return dynamics_.bodyToWheelCommands(last_command_);
}

void MPPIHCCore::setConfig(const ControllerConfig& config)
{
    config_ = config;
    initializeMPPI();
    slip_estimator_ = SlipEstimator(config.slip);
    slip_compensator_ = SlipCompensator(config.slip);
    if (config_.use_sg_filter) {
        initSGFilter();
    }
}

void MPPIHCCore::initSGFilter()
{
    int n = config_.sg_half_window;
    int window_size = 2 * n + 1;
    int poly_order = config_.sg_poly_order;

    Eigen::VectorXd v = Eigen::VectorXd::LinSpaced(window_size, -n, n);
    Eigen::MatrixXd X = Eigen::MatrixXd::Ones(window_size, poly_order + 1);
    for (int i = 1; i <= poly_order; ++i) {
        X.col(i) = X.col(i-1).array() * v.array();
    }

    sg_coeffs_ = (X.transpose() * X).inverse() * X.transpose();
    sg_initialized_ = true;

    // Initialize control history
    for (auto& c : control_history_) {
        c.setZero();
    }
}

Control MPPIHCCore::applyFilter(const ControlSequence& seq)
{
    if (!sg_initialized_) {
        return seq[0];
    }

    int n = config_.sg_half_window;
    int window_size = 2 * n + 1;

    Eigen::Vector3d filtered = Eigen::Vector3d::Zero();
    
    for (int i = 0; i < window_size; ++i) {
        Eigen::Vector3d u;
        if (i < n) {
            u = control_history_[i].toEigen();
        } else {
            int idx = i - n;
            if (idx < T_) {
                u = seq[idx].toEigen();
            } else {
                u = seq[T_-1].toEigen();
            }
        }
        filtered += sg_coeffs_(0, i) * u;
    }

    // Update history (shift and add new)
    for (int i = 0; i < n - 1; ++i) {
        control_history_[i] = control_history_[i + 1];
    }
    Control result;
    result.fromEigen(filtered);
    control_history_[n - 1] = result;

    return result;
}

} // namespace mppi_hc
