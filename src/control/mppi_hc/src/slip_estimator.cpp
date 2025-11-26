/**
 * @file slip_estimator.cpp
 * @brief Implementation of online adaptive slip factor estimator
 */

#include "mppi_hc/slip_estimator.hpp"
#include <numeric>
#include <cmath>
#include <iostream>

namespace mppi_hc
{

SlipEstimator::SlipEstimator(const SlipParams& params)
    : params_(params)
    , slip_factor_(0.0)
{
    last_cmd_.setZero();
}

void SlipEstimator::update(const BodyVelocity& cmd, const BodyVelocity& actual, double dt)
{
    // Check for transient (acceleration) - skip learning during rapid changes
    double dv = std::abs(cmd.vx - last_cmd_.vx) + std::abs(cmd.vy - last_cmd_.vy);
    double dw = std::abs(cmd.omega - last_cmd_.omega);
    last_cmd_ = cmd;

    if (dv > 0.05 || dw > 0.1) {
        // Skip during transients
        return;
    }

    // Low-pass filter actual velocities to reduce noise
    filtered_vx_ = filter_alpha_ * actual.vx + (1.0 - filter_alpha_) * filtered_vx_;
    filtered_vy_ = filter_alpha_ * actual.vy + (1.0 - filter_alpha_) * filtered_vy_;
    filtered_omega_ = filter_alpha_ * actual.omega + (1.0 - filter_alpha_) * filtered_omega_;

    // Compute lateral acceleration demand (centripetal term)
    // a_lat = v_x * omega - this is the driver for slip during cornering
    double a_lat = cmd.vx * cmd.omega;

    // Skip learning when excitation is too low (poor observability)
    if (std::abs(a_lat) < params_.excitation_threshold) {
        return;
    }

    // Compute lateral velocity error (observed slip)
    double vy_error = filtered_vy_ - cmd.vy;

    // Model prediction: vy_slip = -K_slip * a_lat
    double vy_slip_pred = -slip_factor_ * a_lat;

    // Residual (prediction error)
    double residual = vy_error - vy_slip_pred;

    // Gradient descent update
    // J = 0.5 * residual^2
    // dJ/dK = residual * d(residual)/dK = residual * a_lat
    // K_new = K_old - lr * dJ/dK
    slip_factor_ -= params_.learning_rate * residual * a_lat;

    // Clamp to valid range
    slip_factor_ = std::clamp(slip_factor_, params_.slip_factor_min, params_.slip_factor_max);

    // Update statistics
    updateStatistics(residual);

    // Debug output (occasional)
    static int count = 0;
    if (count++ % 50 == 0) {
        std::cout << "[SlipEstimator] K_slip: " << slip_factor_
                  << " | residual: " << residual
                  << " | a_lat: " << a_lat
                  << " | converged: " << (is_converged_ ? "yes" : "no")
                  << std::endl;
    }
}

double SlipEstimator::getEffectiveFriction() const
{
    // mu_eff = mu_base * (1 - 2 * K_slip)
    // Higher slip factor means lower effective friction
    double degradation = 1.0 - 2.0 * slip_factor_;
    degradation = std::max(0.3, degradation);  // Floor to prevent instability
    return params_.base_friction_coeff * degradation;
}

void SlipEstimator::reset()
{
    slip_factor_ = 0.0;
    filtered_vx_ = 0.0;
    filtered_vy_ = 0.0;
    filtered_omega_ = 0.0;
    residual_history_.clear();
    residual_mean_ = 0.0;
    residual_variance_ = 1.0;
    is_converged_ = false;
    last_cmd_.setZero();
}

SlipEstimator::Statistics SlipEstimator::getStatistics() const
{
    Statistics stats;
    stats.current_k_slip = slip_factor_;
    stats.raw_k_slip = slip_factor_;  // In this simple version, same as filtered
    stats.estimation_error = residual_mean_;
    stats.is_converged = is_converged_;
    stats.num_samples = static_cast<int>(residual_history_.size());
    return stats;
}

void SlipEstimator::updateStatistics(double residual)
{
    residual_history_.push_back(residual);
    if (residual_history_.size() > HISTORY_SIZE) {
        residual_history_.pop_front();
    }

    if (residual_history_.size() >= HISTORY_SIZE / 2) {
        // Compute mean
        residual_mean_ = std::accumulate(residual_history_.begin(), 
                                         residual_history_.end(), 0.0) 
                         / residual_history_.size();

        // Compute variance
        double sq_sum = 0.0;
        for (double r : residual_history_) {
            sq_sum += (r - residual_mean_) * (r - residual_mean_);
        }
        residual_variance_ = sq_sum / residual_history_.size();

        // Check convergence: low variance indicates stable estimate
        is_converged_ = (residual_variance_ < 0.01);
    }
}

} // namespace mppi_hc
