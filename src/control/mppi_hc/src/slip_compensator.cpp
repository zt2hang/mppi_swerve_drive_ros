/**
 * @file slip_compensator.cpp
 * @brief Implementation of feedforward slip compensation
 */

#include "mppi_hc/slip_compensator.hpp"
#include <cmath>

namespace mppi_hc
{

SlipCompensator::SlipCompensator(const SlipParams& params)
    : params_(params)
    , gain_(params.compensation_gain)
    , enabled_(params.enable_compensation)
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

BodyVelocity SlipCompensator::computeDelta(const BodyVelocity& planned_cmd, double slip_factor) const
{
    BodyVelocity delta;
    delta.setZero();

    if (!enabled_ || slip_factor < 1e-6) {
        last_compensation_ = delta;
        return delta;
    }

    // Predict expected slip: v_slip = -K_slip * v_x * omega
    double predicted_slip = -slip_factor * planned_cmd.vx * planned_cmd.omega;

    // Feedforward compensation: add opposite velocity to cancel predicted slip
    // Δv_y = -γ * predicted_slip = γ * K_slip * v_x * omega
    delta.vy = -gain_ * predicted_slip;

    // Optional: limit compensation magnitude to avoid over-correction
    double max_compensation = 0.5;  // [m/s]
    delta.vy = std::clamp(delta.vy, -max_compensation, max_compensation);

    last_compensation_ = delta;
    return delta;
}

} // namespace mppi_hc
