#pragma once

/**
 * @file mppi_hc_core.hpp
 * @brief Core MPPI-HC controller (ROS-independent)
 * 
 * Hierarchical Compensated MPPI Architecture:
 * 
 *   Layer 1: MPPI Planning
 *     - Samples trajectories in (vx, vy, omega) space
 *     - Uses slip-aware dynamics for prediction
 *     - Evaluates with slip-aware cost function
 * 
 *   Layer 2: Slip Estimation
 *     - Online gradient descent learning of K_slip
 *     - Updates based on command vs actual velocity
 * 
 *   Layer 3: Slip Compensation
 *     - Feedforward compensation: Δvy = -γ * K_slip * vx * omega
 *     - Active slip cancellation
 */

#include "mppi_hc/types.hpp"
#include "mppi_hc/dynamics.hpp"
#include "mppi_hc/cost_function.hpp"
#include "mppi_hc/slip_estimator.hpp"
#include "mppi_hc/slip_compensator.hpp"

#include <grid_map_core/GridMap.hpp>
#include <random>
#include <Eigen/Dense>

namespace mppi_hc
{

/**
 * @brief Main MPPI-HC controller class
 */
class MPPIHCCore
{
public:
    explicit MPPIHCCore(const ControllerConfig& config);
    ~MPPIHCCore() = default;

    /**
     * @brief Solve MPPI and return optimal control command
     * 
     * @param current_state Current robot state (x, y, yaw)
     * @param collision_map Grid map with collision costs
     * @param distance_error_map Grid map with distance to reference path
     * @param ref_yaw_map Grid map with reference heading
     * @param goal Goal state
     * @return Optimal velocity command (after compensation)
     */
    BodyVelocity solve(
        const State& current_state,
        const grid_map::GridMap& collision_map,
        const grid_map::GridMap& distance_error_map,
        const grid_map::GridMap& ref_yaw_map,
        const State& goal
    );

    /**
     * @brief Solve MPPI with closed-loop feedback compensation
     * 
     * This version uses both feedforward slip compensation AND feedback
     * correction based on actual tracking errors for cm-level accuracy.
     * 
     * @param current_state Current robot state
     * @param collision_map Grid map with collision costs
     * @param distance_error_map Grid map with distance to reference path
     * @param ref_yaw_map Grid map with reference heading
     * @param goal Goal state
     * @param lateral_error Cross-track error (positive = left of path)
     * @param heading_error Heading error (rad)
     * @param path_curvature Local path curvature (1/m)
     * @param dt Time since last update
     * @return Optimal velocity command (with closed-loop compensation)
     */
    BodyVelocity solveWithFeedback(
        const State& current_state,
        const grid_map::GridMap& collision_map,
        const grid_map::GridMap& distance_error_map,
        const grid_map::GridMap& ref_yaw_map,
        const State& goal,
        double lateral_error,
        double heading_error,
        double path_curvature,
        double dt
    );

    /**
     * @brief Update slip estimator with actual measurements
     * Call this after receiving odometry
     */
    void updateEstimator(double actual_vx, double actual_vy, double actual_omega);

    // Get estimator statistics
    SlipEstimator::Statistics getEstimatorStats() const { 
        return slip_estimator_.getStatistics(); 
    }

    // Accessors
    bool isGoalReached() const { return goal_reached_; }
    double getSlipFactor() const { return slip_estimator_.getSlipFactor(); }
    double getCalcTimeMs() const { return calc_time_ms_; }
    double getStateCost() const { return state_cost_; }

    // Get trajectories for visualization
    StateSequence getOptimalTrajectory() const { return optimal_trajectory_; }
    StateTrajectories getEliteTrajectories(int n) const;
    ControlSequence getOptimalControlSequence() const { return optimal_control_seq_; }

    // Get 8-DOF wheel commands
    VehicleCommand8D getWheelCommands() const;

    // Configuration
    void setConfig(const ControllerConfig& config);
    const ControllerConfig& getConfig() const { return config_; }
    
    // Set feedback gains for closed-loop compensation
    void setFeedbackGains(double k_lateral, double k_heading, double k_integral) {
        slip_compensator_.setFeedbackGains(k_lateral, k_heading, k_integral);
    }

private:
    ControllerConfig config_;

    // Core components
    DynamicsModel dynamics_;
    CostFunction cost_function_;
    SlipEstimator slip_estimator_;
    SlipCompensator slip_compensator_;

    // MPPI state
    int K_, T_;  // num_samples, horizon
    std::mt19937 rng_;
    
    // Trajectory storage
    StateTrajectories state_samples_;      // (K, T)
    ControlTrajectories control_samples_;  // (K, T)
    ControlTrajectories noise_samples_;    // (K, T)
    CostVector costs_;                     // (K)
    std::vector<int> cost_ranks_;          // (K) sorted indices
    std::vector<double> weights_;          // (K)
    
    // Optimal solution
    ControlSequence optimal_control_seq_;  // (T)
    StateSequence optimal_trajectory_;     // (T)
    BodyVelocity last_command_;
    
    // Status
    bool goal_reached_ = false;
    double calc_time_ms_ = 0.0;
    double state_cost_ = 0.0;

    // Savitzky-Golay filter
    Eigen::MatrixXd sg_coeffs_;
    ControlSequence control_history_;
    bool sg_initialized_ = false;

    // Methods
    void initializeMPPI();
    void generateNoiseSamples();
    void rolloutTrajectories(const State& initial_state,
                             const grid_map::GridMap& collision_map,
                             const grid_map::GridMap& distance_error_map,
                             const grid_map::GridMap& ref_yaw_map,
                             const State& goal);
    void computeWeights();
    void updateOptimalSequence();
    Control applyFilter(const ControlSequence& seq);
    void initSGFilter();
};

} // namespace mppi_hc
