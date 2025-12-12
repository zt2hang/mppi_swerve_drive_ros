/**
 * @file parameter_estimator.cpp
 * @brief Implementation of online tire parameter estimation
 * 
 * @author ZZT
 * @date 2024
 */

#include "mppi_tf/parameter_estimator.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace mppi_tf
{

// ============================================================================
// Cornering Stiffness Estimator
// ============================================================================

CorneringStiffnessEstimator::CorneringStiffnessEstimator(const EstimatorParams& params)
    : params_(params)
    , C_alpha_(params.C_alpha_min + 0.5 * (params.C_alpha_max - params.C_alpha_min))
    , P_(1000.0)  // Initial covariance (high uncertainty)
    , lambda_(0.98)  // Forgetting factor
    , is_converged_(false)
    , update_count_(0)
{
}

void CorneringStiffnessEstimator::update(
    double slip_angle, double Fy_measured, double Fz, double dt)
{
    // Only update in the linear region (small slip angles)
    if (std::abs(slip_angle) < 0.01 || std::abs(slip_angle) > 0.15) {
        return;  // Skip if slip angle too small (noise) or too large (nonlinear)
    }
    
    if (Fz < 100.0) {
        return;  // Skip if normal load too small
    }
    
    // RLS update for model: Fy = -C_alpha * alpha
    // (negative because Fy opposes slip angle)
    double phi = -slip_angle;  // Regressor
    double y = Fy_measured;    // Measurement
    
    // Prediction error
    double y_pred = C_alpha_ * phi;
    double e = y - y_pred;
    
    // RLS gain
    double denom = lambda_ + phi * P_ * phi;
    if (std::abs(denom) < EPSILON) return;
    
    double K = P_ * phi / denom;
    
    // Update estimate
    double delta = K * e;
    C_alpha_ += delta;
    
    // Bound estimate
    C_alpha_ = std::clamp(C_alpha_, params_.C_alpha_min, params_.C_alpha_max);
    
    // Update covariance
    P_ = (P_ - K * phi * P_) / lambda_;
    
    // Prevent covariance from becoming too small or negative
    P_ = std::max(P_, 1.0);
    
    // Track convergence
    estimate_history_.push_back(C_alpha_);
    if (estimate_history_.size() > HISTORY_SIZE) {
        estimate_history_.pop_front();
    }
    
    update_count_++;
    
    // Check convergence (low variance in recent estimates)
    if (estimate_history_.size() >= HISTORY_SIZE) {
        double mean = std::accumulate(estimate_history_.begin(), 
                                      estimate_history_.end(), 0.0) / estimate_history_.size();
        double variance = 0.0;
        for (double val : estimate_history_) {
            variance += (val - mean) * (val - mean);
        }
        variance /= estimate_history_.size();
        
        is_converged_ = (variance < params_.convergence_threshold * mean * mean);
    }
}

void CorneringStiffnessEstimator::reset()
{
    C_alpha_ = params_.C_alpha_min + 0.5 * (params_.C_alpha_max - params_.C_alpha_min);
    P_ = 1000.0;
    is_converged_ = false;
    update_count_ = 0;
    estimate_history_.clear();
}

// ============================================================================
// Friction Estimator
// ============================================================================

FrictionEstimator::FrictionEstimator(const EstimatorParams& params)
    : params_(params)
    , mu_(0.5 * (params.mu_min + params.mu_max))
    , mu_filtered_(mu_)
    , confidence_(0.0)
    , is_saturated_(false)
    , peak_utilization_(0.0)
    , saturation_threshold_(0.9)
    , saturation_count_(0)
{
}

void FrictionEstimator::update(
    double F_measured, double Fz, double slip_angle, double dt)
{
    if (Fz < 100.0) return;
    
    // Current utilization ratio
    double mu_instant = F_measured / Fz;
    
    // Track utilization history
    utilization_history_.push_back(mu_instant);
    if (utilization_history_.size() > HISTORY_SIZE) {
        utilization_history_.pop_front();
    }
    
    // Find peak utilization (indicates actual friction limit)
    peak_utilization_ = *std::max_element(
        utilization_history_.begin(), utilization_history_.end());
    
    // Saturation detection
    // High utilization + large slip angle = saturated
    bool currently_saturated = (mu_instant > saturation_threshold_) && 
                               (std::abs(slip_angle) > 0.05);
    
    if (currently_saturated) {
        saturation_count_++;
    } else {
        saturation_count_ = std::max(0, saturation_count_ - 1);
    }
    
    is_saturated_ = (saturation_count_ > 5);
    
    // Update friction estimate
    // Only trust measurements near saturation (peak tracking)
    if (is_saturated_ || mu_instant > 0.8 * peak_utilization_) {
        // Weighted update towards measured value
        double alpha = params_.lr_friction;
        mu_ = mu_ + alpha * (mu_instant - mu_);
        
        // Increase confidence when we see saturation
        confidence_ = std::min(1.0, confidence_ + 0.02);
    } else {
        // Slowly decay confidence when not seeing saturation
        confidence_ = std::max(0.0, confidence_ - 0.001);
    }
    
    // Bound estimate
    mu_ = std::clamp(mu_, params_.mu_min, params_.mu_max);
    
    // Low-pass filter
    mu_filtered_ = params_.lpf_alpha * mu_ + (1.0 - params_.lpf_alpha) * mu_filtered_;
}

void FrictionEstimator::reset()
{
    mu_ = 0.5 * (params_.mu_min + params_.mu_max);
    mu_filtered_ = mu_;
    confidence_ = 0.0;
    is_saturated_ = false;
    peak_utilization_ = 0.0;
    saturation_count_ = 0;
    utilization_history_.clear();
}

// ============================================================================
// Combined Tire Parameter Estimator
// ============================================================================

TireParameterEstimator::TireParameterEstimator(
    const EstimatorParams& params,
    const VehicleGeometry& geometry,
    const VehicleMass& mass)
    : params_(params)
    , geometry_(geometry)
    , mass_(mass)
    , cornering_estimators_{
        CorneringStiffnessEstimator(params),
        CorneringStiffnessEstimator(params),
        CorneringStiffnessEstimator(params),
        CorneringStiffnessEstimator(params)
      }
    , friction_estimators_{
        FrictionEstimator(params),
        FrictionEstimator(params),
        FrictionEstimator(params),
        FrictionEstimator(params)
      }
    , C_alpha_vehicle_(params.C_alpha_min + 0.5 * (params.C_alpha_max - params.C_alpha_min))
    , mu_vehicle_(0.5 * (params.mu_min + params.mu_max))
    , is_converged_(false)
    , estimation_enabled_(true)
    , total_updates_(0)
    , velocity_error_integral_(0.0)
    , estimation_error_(0.0)
    , residual_variance_(1.0)
{
    C_alpha_wheels_.fill(C_alpha_vehicle_);
    mu_wheels_.fill(mu_vehicle_);
}

void TireParameterEstimator::update(
    const FullState& state,
    const WheelForces& wheel_forces,
    const Eigen::Vector3d& actual_accel,
    double dt)
{
    if (!estimation_enabled_) return;
    
    // Check excitation
    if (!hasExcitation(state)) return;
    
    // Compute slip angles for each wheel
    auto slip_angles = computeSlipAngles(state);
    
    // Update per-wheel estimators
    for (int i = 0; i < NUM_WHEELS; ++i) {
        WheelIndex idx = static_cast<WheelIndex>(i);
        double Fz = mass_.getStaticWheelLoad(idx);
        
        // TODO: Add load transfer calculation
        // Fz += computeLoadTransfer(actual_accel, idx);
        
        // Update cornering stiffness estimator
        cornering_estimators_[i].update(
            slip_angles[i],
            wheel_forces[i].Fy,
            Fz,
            dt
        );
        
        // Update friction estimator
        double F_total = wheel_forces[i].magnitude();
        friction_estimators_[i].update(
            F_total,
            Fz,
            slip_angles[i],
            dt
        );
        
        C_alpha_wheels_[i] = cornering_estimators_[i].getEstimate();
        mu_wheels_[i] = friction_estimators_[i].getEstimate();
    }
    
    // Fuse wheel-level estimates
    fuseEstimates();
    
    total_updates_++;
}

void TireParameterEstimator::updateFromVelocity(
    const BodyVelocity& cmd_vel,
    const BodyVelocity& actual_vel,
    double dt)
{
    if (!estimation_enabled_) return;
    
    // Need some motion for estimation
    double cmd_speed = cmd_vel.speed();
    if (cmd_speed < params_.min_speed) {
        return;
    }
    
    // Velocity error
    double vy_error = actual_vel.vy - cmd_vel.vy;
    double omega_error = actual_vel.omega - cmd_vel.omega;
    
    // The lateral velocity error during turning indicates slip
    // v_y_error ≈ -K_slip * v_x * omega
    // where K_slip is related to cornering stiffness
    
    double excitation = std::abs(cmd_vel.vx * cmd_vel.omega);
    if (excitation < params_.excitation_threshold) {
        return;
    }
    
    // Simple slip factor estimation (similar to MPPI-HC)
    // K_slip = -v_y_error / (v_x * omega)
    double K_slip_instant = -vy_error / (cmd_vel.vx * cmd_vel.omega + EPSILON);
    K_slip_instant = std::clamp(K_slip_instant, -0.5, 0.5);
    
    // Convert to cornering stiffness change
    // K_slip ≈ L_r * m * v_x / (2 * C_alpha * L) for a simplified model
    // So C_alpha_correction ∝ -K_slip
    double correction = -K_slip_instant * params_.lr_cornering * C_alpha_vehicle_;
    C_alpha_vehicle_ += correction;
    C_alpha_vehicle_ = std::clamp(C_alpha_vehicle_, params_.C_alpha_min, params_.C_alpha_max);
    
    // Update estimation error tracking
    estimation_error_ = std::abs(vy_error);
    
    // Integral term for steady-state accuracy
    velocity_error_integral_ += vy_error * dt;
    velocity_error_integral_ = std::clamp(velocity_error_integral_, -1.0, 1.0);
    
    // Apply integral correction
    double integral_correction = -params_.lr_cornering * 0.1 * velocity_error_integral_ * C_alpha_vehicle_;
    C_alpha_vehicle_ += integral_correction;
    C_alpha_vehicle_ = std::clamp(C_alpha_vehicle_, params_.C_alpha_min, params_.C_alpha_max);
    
    // Update all wheel estimates (simplified: same value)
    C_alpha_wheels_.fill(C_alpha_vehicle_);
    
    // Store for next iteration
    last_cmd_vel_ = cmd_vel;
    last_actual_vel_ = actual_vel;
    
    total_updates_++;
}

void TireParameterEstimator::fuseEstimates()
{
    // Fuse cornering stiffness: weighted average by confidence
    double total_weight = 0.0;
    double weighted_C_alpha = 0.0;
    double weighted_mu = 0.0;
    
    for (int i = 0; i < NUM_WHEELS; ++i) {
        double conf_C = cornering_estimators_[i].getConfidence();
        double conf_mu = friction_estimators_[i].getConfidence();
        
        weighted_C_alpha += conf_C * C_alpha_wheels_[i];
        weighted_mu += conf_mu * mu_wheels_[i];
        total_weight += std::max(conf_C, conf_mu);
    }
    
    if (total_weight > EPSILON) {
        // Blend with current estimate (smoothing)
        double alpha = 0.1;
        double new_C_alpha = weighted_C_alpha / total_weight;
        double new_mu = weighted_mu / total_weight;
        
        C_alpha_vehicle_ = (1.0 - alpha) * C_alpha_vehicle_ + alpha * new_C_alpha;
        mu_vehicle_ = (1.0 - alpha) * mu_vehicle_ + alpha * new_mu;
    }
    
    // Check convergence
    int converged_count = 0;
    for (int i = 0; i < NUM_WHEELS; ++i) {
        if (cornering_estimators_[i].isConverged()) {
            converged_count++;
        }
    }
    is_converged_ = (converged_count >= 2);
}

bool TireParameterEstimator::hasExcitation(const FullState& state) const
{
    // Check minimum speed
    double speed = state.speed();
    if (speed < params_.min_speed) return false;
    
    // Check for lateral motion (excites cornering stiffness)
    double lat_accel = std::abs(state.vx * state.omega);  // Approximate
    if (lat_accel < params_.min_lat_accel) return false;
    
    return true;
}

std::array<double, NUM_WHEELS> TireParameterEstimator::computeSlipAngles(
    const FullState& state) const
{
    std::array<double, NUM_WHEELS> slip_angles;
    BodyVelocity body_vel{state.vx, state.vy, state.omega};
    
    for (int i = 0; i < NUM_WHEELS; ++i) {
        WheelIndex idx = static_cast<WheelIndex>(i);
        Position2D wheel_pos = geometry_.getWheelPosition(idx);
        
        // Wheel velocity in body frame
        BodyVelocity wheel_vel = computeWheelVelocity(body_vel, wheel_pos);
        
        // Slip angle (assuming zero steering for now)
        slip_angles[i] = TireModel::computeSlipAngle(wheel_vel.vx, wheel_vel.vy);
    }
    
    return slip_angles;
}

EstimatorStats TireParameterEstimator::getStatistics() const
{
    EstimatorStats stats;
    stats.C_alpha_est = C_alpha_vehicle_;
    stats.mu_est = mu_vehicle_;
    stats.estimation_error = estimation_error_;
    stats.is_converged = is_converged_;
    stats.num_samples = total_updates_;
    stats.residual_variance = residual_variance_;
    return stats;
}

void TireParameterEstimator::reset()
{
    for (int i = 0; i < NUM_WHEELS; ++i) {
        cornering_estimators_[i].reset();
        friction_estimators_[i].reset();
    }
    
    C_alpha_vehicle_ = params_.C_alpha_min + 0.5 * (params_.C_alpha_max - params_.C_alpha_min);
    mu_vehicle_ = 0.5 * (params_.mu_min + params_.mu_max);
    C_alpha_wheels_.fill(C_alpha_vehicle_);
    mu_wheels_.fill(mu_vehicle_);
    
    is_converged_ = false;
    total_updates_ = 0;
    velocity_error_integral_ = 0.0;
    estimation_error_ = 0.0;
}

void TireParameterEstimator::setManualParams(double C_alpha, double mu)
{
    estimation_enabled_ = false;
    C_alpha_vehicle_ = std::clamp(C_alpha, params_.C_alpha_min, params_.C_alpha_max);
    mu_vehicle_ = std::clamp(mu, params_.mu_min, params_.mu_max);
    C_alpha_wheels_.fill(C_alpha_vehicle_);
    mu_wheels_.fill(mu_vehicle_);
    is_converged_ = true;
}

} // namespace mppi_tf
