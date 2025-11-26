#pragma once

/**
 * @file dynamics.hpp
 * @brief Slip-aware dynamics model for omnidirectional swerve drive
 * 
 * State: (x, y, yaw)
 * Control: (vx, vy, omega)
 * 
 * Dynamics with slip:
 *   v_y_effective = v_y - K_slip * v_x * omega
 *   
 *   ẋ = v_x * cos(yaw) - v_y_eff * sin(yaw)
 *   ẏ = v_x * sin(yaw) + v_y_eff * cos(yaw)
 *   θ̇ = omega
 */

#include "mppi_hc/types.hpp"

namespace mppi_hc
{

/**
 * @brief Slip-aware dynamics model
 */
class DynamicsModel
{
public:
    explicit DynamicsModel(const VehicleParams& params) : params_(params) {}

    /**
     * @brief Compute next state using slip-aware model
     * @param state Current state (x, y, yaw)
     * @param control Control input (vx, vy, omega)
     * @param dt Time step
     * @param slip_factor Estimated slip factor
     * @return Next state
     */
    State step(const State& state, const Control& control, double dt, double slip_factor) const
    {
        // Clamp control inputs
        Control u = control;
        u.clamp(params_.vx_max, params_.vy_max, params_.omega_max);

        // Apply slip model: v_y_effective = v_y - K_slip * v_x * omega
        double vy_slip = -slip_factor * u.vx * u.omega;
        double vy_effective = u.vy + vy_slip;

        // Integrate kinematics
        State next;
        next.x = state.x + u.vx * std::cos(state.yaw) * dt - vy_effective * std::sin(state.yaw) * dt;
        next.y = state.y + u.vx * std::sin(state.yaw) * dt + vy_effective * std::cos(state.yaw) * dt;
        next.yaw = state.yaw + u.omega * dt;
        next.normalizeYaw();

        return next;
    }

    /**
     * @brief Convert body velocity to 8-DOF wheel commands
     */
    VehicleCommand8D bodyToWheelCommands(const BodyVelocity& body_vel) const
    {
        VehicleCommand8D cmd;
        
        // Front-left
        double vx_fl = body_vel.vx - params_.d_l * body_vel.omega;
        double vy_fl = body_vel.vy + params_.l_f * body_vel.omega;
        cmd.fl.velocity = std::sqrt(vx_fl * vx_fl + vy_fl * vy_fl);
        cmd.fl.steer = std::atan2(vy_fl, vx_fl);

        // Front-right
        double vx_fr = body_vel.vx + params_.d_r * body_vel.omega;
        double vy_fr = body_vel.vy + params_.l_f * body_vel.omega;
        cmd.fr.velocity = std::sqrt(vx_fr * vx_fr + vy_fr * vy_fr);
        cmd.fr.steer = std::atan2(vy_fr, vx_fr);

        // Rear-left
        double vx_rl = body_vel.vx - params_.d_l * body_vel.omega;
        double vy_rl = body_vel.vy - params_.l_r * body_vel.omega;
        cmd.rl.velocity = std::sqrt(vx_rl * vx_rl + vy_rl * vy_rl);
        cmd.rl.steer = std::atan2(vy_rl, vx_rl);

        // Rear-right
        double vx_rr = body_vel.vx + params_.d_r * body_vel.omega;
        double vy_rr = body_vel.vy - params_.l_r * body_vel.omega;
        cmd.rr.velocity = std::sqrt(vx_rr * vx_rr + vy_rr * vy_rr);
        cmd.rr.steer = std::atan2(vy_rr, vx_rr);

        return cmd;
    }

    /**
     * @brief Predict lateral slip velocity
     */
    double predictSlipVelocity(const Control& control, double slip_factor) const
    {
        return -slip_factor * control.vx * control.omega;
    }

    const VehicleParams& getParams() const { return params_; }

private:
    VehicleParams params_;
};

} // namespace mppi_hc
