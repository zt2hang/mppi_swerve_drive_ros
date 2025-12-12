#pragma once

/**
 * @file mppi_tf_core.hpp
 * @brief Core MPPI-TireForce controller (ROS-independent)
 * 
 * This is the main MPPI controller class that implements tire force-based
 * sampling and optimization. The architecture is hierarchical:
 * 
 * ┌────────────────────────────────────────────────────────────────┐
 * │                    MPPI-TireForce Architecture                 │
 * ├────────────────────────────────────────────────────────────────┤
 * │  Layer 1: MPPI Sampling (Force or Velocity Space)              │
 * │     - Generate K trajectory samples                            │
 * │     - Propagate dynamics with tire model                       │
 * │     - Evaluate tire-aware cost function                        │
 * │                                                                │
 * │  Layer 2: Force Allocation                                     │
 * │     - Distribute body force to 4 wheels                        │
 * │     - Respect friction circle constraints                      │
 * │     - Optimize for friction margin                             │
 * │                                                                │
 * │  Layer 3: Parameter Estimation                                 │
 * │     - Online cornering stiffness estimation                    │
 * │     - Online friction coefficient estimation                   │
 * │     - Adaptive model update                                    │
 * │                                                                │
 * │  Layer 4: Command Generation                                   │
 * │     - Convert forces to wheel commands                         │
 * │     - Apply smoothing filter                                   │
 * │     - Output 8-DOF vehicle command                             │
 * └────────────────────────────────────────────────────────────────┘
 * 
 * Key Features:
 * - Dual-mode sampling: Force-space or Velocity-space
 * - Physics-based tire model (Pacejka Magic Formula)
 * - Friction-aware cost function
 * - Online parameter adaptation
 * - Closed-loop feedback compensation
 * 
 * @author ZZT
 * @date 2024
 */

#include "mppi_tf/types.hpp"
#include "mppi_tf/dynamics.hpp"
#include "mppi_tf/tire_model.hpp"
#include "mppi_tf/force_allocator.hpp"
#include "mppi_tf/parameter_estimator.hpp"
#include "mppi_tf/cost_function.hpp"

#include <grid_map_core/GridMap.hpp>
#include <random>
#include <Eigen/Dense>

namespace mppi_tf
{

/**
 * @brief Sampling mode for MPPI
 */
enum class SamplingMode
{
    FORCE_SPACE,    // Sample in (Fx, Fy, Mz) space
    VELOCITY_SPACE, // Sample in (vx, vy, omega) space
    HYBRID          // Force-space with velocity fallback
};

/**
 * @brief Main MPPI-TireForce controller class
 */
class MPPITFCore
{
public:
    /**
     * @brief Construct controller with configuration
     */
    explicit MPPITFCore(const ControllerConfig& config);
    ~MPPITFCore() = default;

    // ========================================================================
    // Main Interface
    // ========================================================================

    /**
     * @brief Solve MPPI and return optimal body velocity command
     * 
     * This is the primary interface. It:
     * 1. Samples trajectories in force/velocity space
     * 2. Evaluates tire-aware costs
     * 3. Computes optimal control via importance sampling
     * 4. Converts to velocity command
     * 
     * @param current_state Current robot state (pose + velocity)
     * @param collision_map Grid map with collision costs
     * @param distance_map Grid map with distance to reference path
     * @param yaw_map Grid map with reference heading
     * @param goal Goal state
     * @return Optimal body velocity command
     */
    BodyVelocity solve(
        const FullState& current_state,
        const grid_map::GridMap& collision_map,
        const grid_map::GridMap& distance_map,
        const grid_map::GridMap& yaw_map,
        const FullState& goal
    );

    /**
     * @brief Solve with closed-loop feedback compensation
     * 
     * Adds feedback correction for tracking errors.
     * 
     * @param lateral_error Cross-track error [m]
     * @param heading_error Heading error [rad]
     * @param path_curvature Local path curvature [1/m]
     * @param dt Time since last update
     */
    BodyVelocity solveWithFeedback(
        const FullState& current_state,
        const grid_map::GridMap& collision_map,
        const grid_map::GridMap& distance_map,
        const grid_map::GridMap& yaw_map,
        const FullState& goal,
        double lateral_error,
        double heading_error,
        double path_curvature,
        double dt
    );

    // ========================================================================
    // Estimator Interface
    // ========================================================================

    /**
     * @brief Update parameter estimator with actual measurements
     * 
     * Call this after receiving odometry to update tire parameter estimates.
     */
    void updateEstimator(
        const BodyVelocity& cmd_vel,
        const BodyVelocity& actual_vel,
        double dt
    );

    /**
     * @brief Get estimator statistics
     */
    EstimatorStats getEstimatorStats() const;

    /**
     * @brief Reset estimator
     */
    void resetEstimator() { estimator_.reset(); }

    // ========================================================================
    // Configuration
    // ========================================================================

    void setConfig(const ControllerConfig& config);
    const ControllerConfig& getConfig() const { return config_; }

    void setSamplingMode(SamplingMode mode) { sampling_mode_ = mode; }
    SamplingMode getSamplingMode() const { return sampling_mode_; }

    void setFeedbackGains(double k_lat, double k_head, double k_integral);

    // ========================================================================
    // Accessors
    // ========================================================================

    bool isGoalReached() const { return goal_reached_; }
    double getCalcTimeMs() const { return calc_time_ms_; }
    double getStateCost() const { return state_cost_; }

    // Tire parameters
    double getCorneringStiffness() const { return estimator_.getCorneringStiffness(); }
    double getFrictionCoeff() const { return estimator_.getFrictionCoeff(); }

    // Trajectory access (for visualization)
    StateSequence getOptimalTrajectory() const { return optimal_trajectory_; }
    StateTrajectories getEliteTrajectories(int n) const;
    ControlSequence getOptimalForceSequence() const { return optimal_force_seq_; }
    VelocitySequence getOptimalVelocitySequence() const { return optimal_velocity_seq_; }

    // 8-DOF wheel commands
    VehicleCommand8D getWheelCommands() const;

    // Force allocation result
    AllocationResult getLastAllocationResult() const { return last_allocation_; }

private:
    ControllerConfig config_;
    SamplingMode sampling_mode_ = SamplingMode::VELOCITY_SPACE;

    // Core components
    DynamicsModel dynamics_;
    TireModel tire_model_;
    ForceAllocator force_allocator_;
    TireParameterEstimator estimator_;
    CostFunction cost_function_;

    // MPPI parameters
    int K_;  // Number of samples
    int T_;  // Prediction horizon

    // Random number generation
    std::mt19937 rng_;

    // Sample storage
    StateTrajectories state_samples_;           // (K, T)
    ControlTrajectories force_samples_;         // (K, T) - force-space
    std::vector<VelocitySequence> vel_samples_; // (K, T) - velocity-space
    ControlTrajectories noise_force_;           // (K, T)
    std::vector<VelocitySequence> noise_vel_;   // (K, T)
    CostVector costs_;                          // (K)
    std::vector<int> cost_ranks_;               // (K)
    std::vector<double> weights_;               // (K)

    // Optimal solution
    StateSequence optimal_trajectory_;
    ControlSequence optimal_force_seq_;
    VelocitySequence optimal_velocity_seq_;
    BodyVelocity last_command_;
    BodyForce last_force_;

    // State
    bool goal_reached_ = false;
    double calc_time_ms_ = 0.0;
    double state_cost_ = 0.0;
    AllocationResult last_allocation_;

    // Feedback control
    double k_lateral_ = 1.5;
    double k_heading_ = 0.5;
    double k_integral_ = 0.2;
    double error_integral_ = 0.0;

    // Savitzky-Golay filter
    Eigen::MatrixXd sg_coeffs_;
    VelocitySequence velocity_history_;

    // ========================================================================
    // Internal Methods
    // ========================================================================

    void initializeMPPI();
    void initSGFilter();

    /**
     * @brief Generate noise samples
     */
    void generateNoiseSamples();

    /**
     * @brief Rollout trajectories with force-space sampling
     */
    void rolloutTrajectoriesForce(
        const FullState& initial,
        const grid_map::GridMap& collision_map,
        const grid_map::GridMap& distance_map,
        const grid_map::GridMap& yaw_map,
        const FullState& goal
    );

    /**
     * @brief Rollout trajectories with velocity-space sampling
     */
    void rolloutTrajectoriesVelocity(
        const FullState& initial,
        const grid_map::GridMap& collision_map,
        const grid_map::GridMap& distance_map,
        const grid_map::GridMap& yaw_map,
        const FullState& goal
    );

    /**
     * @brief Compute importance weights from costs
     */
    void computeWeights();

    /**
     * @brief Update optimal control sequence from weighted samples
     */
    void updateOptimalSequence();

    /**
     * @brief Apply Savitzky-Golay filter to smooth command
     */
    BodyVelocity applySGFilter(const VelocitySequence& seq);

    /**
     * @brief Convert force command to velocity command
     */
    BodyVelocity forceToVelocity(const BodyForce& force, const FullState& state, double dt);

    /**
     * @brief Apply closed-loop feedback compensation
     */
    BodyVelocity applyFeedback(
        const BodyVelocity& planned,
        double lateral_error,
        double heading_error,
        double path_curvature,
        double dt
    );

    /**
     * @brief Check goal reached condition
     */
    bool checkGoalReached(const FullState& state, const FullState& goal);
};

} // namespace mppi_tf
