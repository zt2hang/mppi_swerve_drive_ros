#pragma once

/**
 * @file slip_compensator.hpp
 * @brief Closed-loop slip compensation layer (Feedforward + Feedback)
 * 
 * Provides active slip compensation through two mechanisms:
 * 
 * 1. FEEDFORWARD: Pre-compensate predicted slip based on model
 *    Δv_y_ff = -γ * K_slip * v_x * omega
 * 
 * 2. FEEDBACK: Correct residual error based on actual tracking performance
 *    Δv_y_fb = -K_fb * e_lateral
 * 
 * Combined compensation:
 *    Δv_y = Δv_y_ff + Δv_y_fb
 * 
 * This closed-loop architecture ensures:
 *   - Fast response to predicted disturbances (feedforward)
 *   - Zero steady-state error through feedback correction
 *   - Adaptation to model uncertainty
 */

#include "mppi_hc/types.hpp"
#include "mppi_hc/slip_estimator.hpp"
#include <deque>

namespace mppi_hc
{

/**
 * @brief Closed-loop slip compensator with feedforward + feedback
 */
class SlipCompensator
{
public:
    explicit SlipCompensator(const SlipParams& params);

    /**
     * @brief Compute compensated velocity command (feedforward only, legacy)
     */
    BodyVelocity compensate(const BodyVelocity& planned_cmd, double slip_factor) const;

    /**
     * @brief Compute compensated velocity command with closed-loop feedback
     * @param planned_cmd Command from MPPI planning layer
     * @param slip_factor Current estimated slip factor
     * @param lateral_error Current cross-track error (positive = robot left of path)
     * @param heading_error Current heading error (rad)
     * @param path_curvature Local path curvature (1/m)
     * @return Compensated velocity command
     */
    BodyVelocity compensateClosedLoop(
        const BodyVelocity& planned_cmd,
        double slip_factor,
        double lateral_error,
        double heading_error,
        double path_curvature
    );

    /**
     * @brief Update tracking error for integral term
     */
    void updateError(double lateral_error, double dt);

    /**
     * @brief Reset integrator
     */
    void resetIntegrator();

    /**
     * @brief Decay integrator by a factor (for near-goal behavior)
     * @param factor Decay factor [0, 1]: 0 = full reset, 1 = no change
     */
    void decayIntegrator(double factor);

    /**
     * @brief Compute compensation delta only
     */
    BodyVelocity computeDelta(const BodyVelocity& planned_cmd, double slip_factor) const;

    // Getters/Setters
    void setGain(double gain) { gain_ = gain; }
    double getGain() const { return gain_; }
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }
    BodyVelocity getLastCompensation() const { return last_compensation_; }
    
    // Feedback gains
    void setFeedbackGains(double k_lateral, double k_heading, double k_integral) {
        k_lateral_ = k_lateral;
        k_heading_ = k_heading;
        k_integral_ = k_integral;
    }
    
    // Statistics
    double getFeedforwardComponent() const { return ff_component_; }
    double getFeedbackComponent() const { return fb_component_; }
    double getIntegralComponent() const { return integral_component_; }

private:
    SlipParams params_;
    double gain_;       // Feedforward gain γ
    bool enabled_;
    
    // Feedback control gains
    double k_lateral_;      // Lateral error feedback gain
    double k_heading_;      // Heading error feedback gain  
    double k_integral_;     // Integral gain for steady-state error
    
    // Integrator state
    double error_integral_;
    double integral_limit_;
    
    // Low-pass filter for error derivative
    double prev_lateral_error_;
    double error_derivative_;
    double derivative_alpha_;
    
    // Statistics
    mutable double ff_component_;
    mutable double fb_component_;
    mutable double integral_component_;
    mutable BodyVelocity last_compensation_;
};

} // namespace mppi_hc
