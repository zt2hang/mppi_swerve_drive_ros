#pragma once

/**
 * @file slip_compensator.hpp
 * @brief Feedforward slip compensation layer
 * 
 * Provides active slip compensation by predicting and pre-compensating
 * the expected lateral slip before it occurs.
 * 
 * Compensation law:
 *   Δv_y = -γ * K_slip * v_x * omega
 * 
 * This forms a dual defense mechanism:
 *   1. Passive (cost function): Penalize high slip risk inputs
 *   2. Active (feedforward): Pre-compensate predicted slip
 */

#include "mppi_hc/types.hpp"
#include "mppi_hc/slip_estimator.hpp"

namespace mppi_hc
{

/**
 * @brief Feedforward slip compensator
 * 
 * Takes the MPPI planned command and adds compensation based on
 * the estimated slip factor to counteract predicted lateral slip.
 */
class SlipCompensator
{
public:
    explicit SlipCompensator(const SlipParams& params);

    /**
     * @brief Compute compensated velocity command
     * @param planned_cmd Command from MPPI planning layer
     * @param slip_factor Current estimated slip factor
     * @return Compensated velocity command
     */
    BodyVelocity compensate(const BodyVelocity& planned_cmd, double slip_factor) const;

    /**
     * @brief Compute compensation delta only
     * @param planned_cmd Command from MPPI planning layer
     * @param slip_factor Current estimated slip factor
     * @return Compensation delta (Δvx, Δvy, Δomega)
     */
    BodyVelocity computeDelta(const BodyVelocity& planned_cmd, double slip_factor) const;

    /**
     * @brief Set compensation gain
     */
    void setGain(double gain) { gain_ = gain; }
    double getGain() const { return gain_; }

    /**
     * @brief Enable/disable compensation
     */
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }

    /**
     * @brief Get last compensation applied
     */
    BodyVelocity getLastCompensation() const { return last_compensation_; }

private:
    SlipParams params_;
    double gain_;
    bool enabled_;
    mutable BodyVelocity last_compensation_;
};

} // namespace mppi_hc
