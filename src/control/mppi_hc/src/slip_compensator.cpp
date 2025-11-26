/**
 * @file slip_compensator.cpp
 * @brief Implementation of closed-loop slip compensation (FF + FB)
 */

#include "mppi_hc/slip_compensator.hpp"
#include <cmath>
#include <algorithm>

namespace mppi_hc
{

SlipCompensator::SlipCompensator(const SlipParams& params)
    : params_(params)
    , gain_(params.compensation_gain)
    , enabled_(params.enable_compensation)
    , k_lateral_(1.5)       // Reduced lateral error gain
    , k_heading_(0.5)       // Reduced heading error gain
    , k_integral_(0.2)      // Reduced integral gain
    , error_integral_(0.0)
    , integral_limit_(0.15) // Reduced max integral contribution [m/s]
    , prev_lateral_error_(0.0)
    , error_derivative_(0.0)
    , derivative_alpha_(0.15)  // More smoothing
    , ff_component_(0.0)
    , fb_component_(0.0)
    , integral_component_(0.0)
{
    last_compensation_.setZero();
}

BodyVelocity SlipCompensator::compensate(const BodyVelocity& planned_cmd, double slip_factor) const
{
    if (!enabled_) {
        last_compensation_.setZero();
        return planned_cmd;
    }

    BodyVelocity delta = computeDelta(planned_cmd, slip_factor);
    
    BodyVelocity result;
    result.vx = planned_cmd.vx + delta.vx;
    result.vy = planned_cmd.vy + delta.vy;
    result.omega = planned_cmd.omega + delta.omega;

    return result;
}

BodyVelocity SlipCompensator::compensateClosedLoop(
    const BodyVelocity& planned_cmd,
    double slip_factor,
    double lateral_error,
    double heading_error,
    double path_curvature
)
{
    if (!enabled_) {
        last_compensation_.setZero();
        ff_component_ = 0.0;
        fb_component_ = 0.0;
        integral_component_ = 0.0;
        return planned_cmd;
    }

    BodyVelocity delta;
    delta.setZero();

    // =========================================================================
    // 1. FEEDFORWARD COMPENSATION (model-based prediction)
    // =========================================================================
    // Predict slip: v_slip = -K_slip * v_x * omega
    double predicted_slip = -slip_factor * planned_cmd.vx * planned_cmd.omega;
    // Feedforward: cancel predicted slip
    double delta_vy_ff = -gain_ * predicted_slip;
    
    // Scale feedforward based on curvature (only active during turns)
    double abs_curvature = std::abs(path_curvature);
    double curvature_factor = std::min(1.0, abs_curvature * 5.0);  // Ramps up from 0 to 1
    delta_vy_ff *= curvature_factor;

    // =========================================================================
    // 2. FEEDBACK COMPENSATION (error-based correction)
    // =========================================================================
    // Apply deadzone to prevent oscillation on small errors
    const double deadzone = 0.03;  // 3cm deadzone (increased)
    double effective_lateral_error = lateral_error;
    if (std::abs(lateral_error) < deadzone) {
        effective_lateral_error = 0.0;
    } else {
        effective_lateral_error = lateral_error - std::copysign(deadzone, lateral_error);
    }
    
    // CRITICAL: Scale down feedback gains on straight sections (low curvature)
    // This prevents oscillation after turns
    double feedback_scale = std::min(1.0, abs_curvature * 3.0 + 0.3);  // Min 0.3, max 1.0
    
    // Proportional term: lateral error correction
    // Positive error = robot is to the left → need negative vy to go right
    double delta_vy_p = -k_lateral_ * effective_lateral_error * feedback_scale;
    
    // Integral term: accumulate steady-state error (only when significant error exists)
    double delta_vy_i = -k_integral_ * error_integral_ * feedback_scale;
    delta_vy_i = std::clamp(delta_vy_i, -integral_limit_, integral_limit_);
    
    // Heading correction: align with path direction
    // Scale down on straight sections to prevent oscillation
    double heading_scale = std::min(1.0, abs_curvature * 2.0 + 0.2);
    double delta_omega_fb = -k_heading_ * heading_error * heading_scale;
    
    // Derivative term (for damping, using filtered derivative)
    double delta_vy_d = -0.15 * error_derivative_ * feedback_scale;

    // =========================================================================
    // 3. COMBINE FEEDFORWARD + FEEDBACK
    // =========================================================================
    delta.vy = delta_vy_ff + delta_vy_p + delta_vy_i + delta_vy_d;
    delta.omega = delta_omega_fb;
    delta.vx = 0.0;  // No longitudinal compensation

    // Limit total compensation to avoid instability
    const double max_vy_compensation = 0.8;  // [m/s]
    const double max_omega_compensation = 0.5;  // [rad/s]
    delta.vy = std::clamp(delta.vy, -max_vy_compensation, max_vy_compensation);
    delta.omega = std::clamp(delta.omega, -max_omega_compensation, max_omega_compensation);

    // Store statistics
    ff_component_ = delta_vy_ff;
    fb_component_ = delta_vy_p + delta_vy_d;
    integral_component_ = delta_vy_i;
    last_compensation_ = delta;

    // Apply compensation
    BodyVelocity result;
    result.vx = planned_cmd.vx;
    result.vy = planned_cmd.vy + delta.vy;
    result.omega = planned_cmd.omega + delta.omega;

    return result;
}

void SlipCompensator::updateError(double lateral_error, double dt)
{
    if (dt < 0.001) return;
    
    // Update derivative with low-pass filter
    double raw_derivative = (lateral_error - prev_lateral_error_) / dt;
    error_derivative_ = derivative_alpha_ * raw_derivative + 
                       (1.0 - derivative_alpha_) * error_derivative_;
    prev_lateral_error_ = lateral_error;
    
    // Update integral with anti-windup
    const double deadzone = 0.03;  // Match deadzone in compensateClosedLoop
    if (std::abs(lateral_error) > deadzone) {
        error_integral_ += lateral_error * dt;
        // Anti-windup: clamp integral
        double max_integral = integral_limit_ / std::max(k_integral_, 0.01);
        error_integral_ = std::clamp(error_integral_, -max_integral, max_integral);
    } else {
        // Aggressive decay when in deadzone (prevents lingering oscillation)
        error_integral_ *= 0.85;
    }
    
    // Also decay integral when error derivative is opposite sign (error improving)
    if (lateral_error * error_derivative_ < 0) {
        error_integral_ *= 0.9;  // Error is reducing, decay integral faster
    }
}

void SlipCompensator::resetIntegrator()
{
    error_integral_ = 0.0;
    error_derivative_ = 0.0;
    prev_lateral_error_ = 0.0;
}

void SlipCompensator::decayIntegrator(double factor)
{
    factor = std::clamp(factor, 0.0, 1.0);
    error_integral_ *= factor;
    error_derivative_ *= factor;
}

BodyVelocity SlipCompensator::computeDelta(const BodyVelocity& planned_cmd, double slip_factor) const
{
    BodyVelocity delta;
    delta.setZero();

    if (!enabled_ || slip_factor < 1e-6) {
        last_compensation_ = delta;
        return delta;
    }

    double predicted_slip = -slip_factor * planned_cmd.vx * planned_cmd.omega;
    delta.vy = -gain_ * predicted_slip;

    double max_compensation = 0.5;
    delta.vy = std::clamp(delta.vy, -max_compensation, max_compensation);

    last_compensation_ = delta;
    ff_component_ = delta.vy;
    fb_component_ = 0.0;
    integral_component_ = 0.0;
    
    return delta;
}

} // namespace mppi_hc
