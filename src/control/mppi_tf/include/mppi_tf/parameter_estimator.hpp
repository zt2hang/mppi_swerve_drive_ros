#pragma once

/**
 * @file parameter_estimator.hpp
 * @brief Online adaptive estimation of tire and road parameters
 * 
 * This module implements real-time estimation of critical tire-road parameters:
 * 1. Cornering stiffness (C_alpha) - how aggressively the tire generates lateral force
 * 2. Friction coefficient (mu) - peak friction available
 * 
 * Estimation Methods:
 * ==================
 * 
 * 1. Cornering Stiffness Estimation:
 *    - Uses recursive least squares (RLS) on the linear region
 *    - Model: Fy = C_alpha * alpha (for small slip angles)
 *    - Requires sufficient excitation (turning motion)
 * 
 * 2. Friction Coefficient Estimation:
 *    - Monitors force utilization and detects saturation
 *    - When tire saturates: mu ≈ F_measured / Fz
 *    - Uses moving window peak detection
 * 
 * 3. Combined Estimator:
 *    - Fuses cornering stiffness and friction estimates
 *    - Accounts for load transfer effects
 *    - Provides confidence metrics
 * 
 * Features:
 * - Forgetting factor for adaptation to changing conditions
 * - Excitation detection to avoid updates on bad data
 * - Bounded parameter estimates for safety
 * - Convergence detection
 * 
 * @author ZZT
 * @date 2024
 */

#include "mppi_tf/types.hpp"
#include "mppi_tf/tire_model.hpp"
#include <deque>
#include <Eigen/Dense>

namespace mppi_tf
{

/**
 * @brief Recursive Least Squares estimator for cornering stiffness
 */
class CorneringStiffnessEstimator
{
public:
    explicit CorneringStiffnessEstimator(const EstimatorParams& params);

    /**
     * @brief Update estimate with new measurement
     * @param slip_angle Measured slip angle [rad]
     * @param Fy_measured Measured lateral force [N]
     * @param Fz Normal load [N]
     * @param dt Time step [s]
     */
    void update(double slip_angle, double Fy_measured, double Fz, double dt);

    /**
     * @brief Get current cornering stiffness estimate
     */
    double getEstimate() const { return C_alpha_; }

    /**
     * @brief Get estimation confidence (inverse of covariance)
     */
    double getConfidence() const { return 1.0 / (P_ + EPSILON); }

    /**
     * @brief Check if estimate has converged
     */
    bool isConverged() const { return is_converged_; }

    /**
     * @brief Reset estimator to initial state
     */
    void reset();

private:
    EstimatorParams params_;
    
    // RLS state
    double C_alpha_;        // Current estimate [N/rad]
    double P_;              // Estimation covariance
    double lambda_;         // Forgetting factor (0.95-0.99)
    
    // Convergence tracking
    std::deque<double> estimate_history_;
    bool is_converged_;
    int update_count_;
    
    static constexpr int HISTORY_SIZE = 50;
};

/**
 * @brief Friction coefficient estimator using saturation detection
 */
class FrictionEstimator
{
public:
    explicit FrictionEstimator(const EstimatorParams& params);

    /**
     * @brief Update estimate with force measurements
     * @param F_measured Measured total tire force magnitude [N]
     * @param Fz Normal load [N]
     * @param slip_angle Current slip angle [rad]
     * @param dt Time step [s]
     */
    void update(double F_measured, double Fz, double slip_angle, double dt);

    /**
     * @brief Get current friction coefficient estimate
     */
    double getEstimate() const { return mu_; }

    /**
     * @brief Get confidence level
     */
    double getConfidence() const { return confidence_; }

    /**
     * @brief Check if tire is currently saturated
     */
    bool isSaturated() const { return is_saturated_; }

    /**
     * @brief Reset estimator
     */
    void reset();

private:
    EstimatorParams params_;
    
    // Estimate state
    double mu_;             // Current friction estimate
    double mu_filtered_;    // Low-pass filtered estimate
    double confidence_;     // Estimation confidence [0, 1]
    bool is_saturated_;     // Current saturation state
    
    // Peak tracking
    std::deque<double> utilization_history_;
    double peak_utilization_;
    
    // Saturation detection
    double saturation_threshold_;  // Utilization above this = saturated
    int saturation_count_;
    
    static constexpr int HISTORY_SIZE = 100;
};

/**
 * @brief Combined tire parameter estimator
 * 
 * Fuses cornering stiffness and friction estimates from multiple wheels
 * to provide robust vehicle-level parameter estimates.
 */
class TireParameterEstimator
{
public:
    explicit TireParameterEstimator(
        const EstimatorParams& params,
        const VehicleGeometry& geometry,
        const VehicleMass& mass
    );

    /**
     * @brief Update estimator with vehicle state and measured forces
     * 
     * @param state Current vehicle state (includes velocities)
     * @param wheel_forces Measured/commanded wheel forces
     * @param actual_accel Measured body acceleration (from IMU/odom)
     * @param dt Time step [s]
     */
    void update(
        const FullState& state,
        const WheelForces& wheel_forces,
        const Eigen::Vector3d& actual_accel,
        double dt
    );

    /**
     * @brief Simplified update using body velocities only
     * 
     * Estimates parameters from the discrepancy between commanded
     * and actual body velocities.
     * 
     * @param cmd_vel Commanded body velocity
     * @param actual_vel Actual body velocity (from odometry)
     * @param dt Time step
     */
    void updateFromVelocity(
        const BodyVelocity& cmd_vel,
        const BodyVelocity& actual_vel,
        double dt
    );

    // ========================================================================
    // Getters
    // ========================================================================

    /**
     * @brief Get estimated cornering stiffness
     */
    double getCorneringStiffness() const { return C_alpha_vehicle_; }

    /**
     * @brief Get estimated friction coefficient
     */
    double getFrictionCoeff() const { return mu_vehicle_; }

    /**
     * @brief Get per-wheel cornering stiffness estimates
     */
    const std::array<double, NUM_WHEELS>& getWheelCorneringStiffness() const {
        return C_alpha_wheels_;
    }

    /**
     * @brief Get estimator statistics for monitoring
     */
    EstimatorStats getStatistics() const;

    /**
     * @brief Check if estimator has converged
     */
    bool isConverged() const { return is_converged_; }

    /**
     * @brief Reset all estimators
     */
    void reset();

    /**
     * @brief Set manual parameter override (disables estimation)
     */
    void setManualParams(double C_alpha, double mu);

    /**
     * @brief Enable/disable estimation
     */
    void setEstimationEnabled(bool enabled) { estimation_enabled_ = enabled; }

private:
    EstimatorParams params_;
    VehicleGeometry geometry_;
    VehicleMass mass_;
    
    // Per-wheel estimators
    std::array<CorneringStiffnessEstimator, NUM_WHEELS> cornering_estimators_;
    std::array<FrictionEstimator, NUM_WHEELS> friction_estimators_;
    
    // Vehicle-level estimates (fused from wheels)
    double C_alpha_vehicle_;
    double mu_vehicle_;
    std::array<double, NUM_WHEELS> C_alpha_wheels_;
    std::array<double, NUM_WHEELS> mu_wheels_;
    
    // State
    bool is_converged_;
    bool estimation_enabled_;
    int total_updates_;
    
    // Velocity-based estimation state
    BodyVelocity last_cmd_vel_;
    BodyVelocity last_actual_vel_;
    double velocity_error_integral_;
    
    // Statistics
    double estimation_error_;
    double residual_variance_;

    /**
     * @brief Fuse per-wheel estimates to vehicle-level estimate
     */
    void fuseEstimates();

    /**
     * @brief Check excitation conditions
     */
    bool hasExcitation(const FullState& state) const;
    
    /**
     * @brief Compute slip angles for all wheels
     */
    std::array<double, NUM_WHEELS> computeSlipAngles(const FullState& state) const;
};

} // namespace mppi_tf
