#pragma once

/**
 * @file slip_estimator.hpp
 * @brief Online adaptive slip factor estimator
 * 
 * Estimates the slip factor K_slip using gradient descent:
 *   v_slip = -K_slip * v_x * omega
 * 
 * The estimator learns from the difference between commanded and actual velocities.
 */

#include "mppi_hc/types.hpp"
#include <deque>

namespace mppi_hc
{

/**
 * @brief Online adaptive estimator for tire slip factor
 * 
 * Uses gradient descent to minimize the prediction error:
 *   J = 0.5 * (v_y_actual - v_y_predicted)^2
 * 
 * Where v_y_predicted = v_y_cmd - K_slip * v_x * omega
 */
class SlipEstimator
{
public:
    /**
     * @brief Statistics structure for external monitoring
     */
    struct Statistics
    {
        double current_k_slip = 0.0;
        double raw_k_slip = 0.0;
        double estimation_error = 0.0;
        bool is_converged = false;
        int num_samples = 0;
    };

    explicit SlipEstimator(const SlipParams& params);

    /**
     * @brief Update estimator with new measurement
     * @param cmd Commanded body velocity
     * @param actual Actual body velocity (from odometry)
     * @param dt Time step
     */
    void update(const BodyVelocity& cmd, const BodyVelocity& actual, double dt);

    /**
     * @brief Get current estimated slip factor
     */
    double getSlipFactor() const { return slip_factor_; }

    /**
     * @brief Get effective friction coefficient
     * mu_eff = mu_base * (1 - 2 * slip_factor)
     */
    double getEffectiveFriction() const;

    /**
     * @brief Get statistics for monitoring
     */
    Statistics getStatistics() const;

    /**
     * @brief Reset estimator to initial state
     */
    void reset();

    /**
     * @brief Check if estimator has converged (low variance)
     */
    bool isConverged() const { return is_converged_; }

    // Statistics
    double getResidualMean() const { return residual_mean_; }
    double getResidualVariance() const { return residual_variance_; }

private:
    SlipParams params_;
    
    // Estimated parameter
    double slip_factor_ = 0.0;
    
    // State tracking
    BodyVelocity last_cmd_;
    bool is_converged_ = false;
    
    // Low-pass filtered actual velocities (for noise reduction)
    double filtered_vx_ = 0.0;
    double filtered_vy_ = 0.0;
    double filtered_omega_ = 0.0;
    double filter_alpha_ = 0.2;  // Low-pass filter coefficient
    
    // Statistics for convergence detection
    std::deque<double> residual_history_;
    static constexpr size_t HISTORY_SIZE = 50;
    double residual_mean_ = 0.0;
    double residual_variance_ = 1.0;

    void updateStatistics(double residual);
};

} // namespace mppi_hc
