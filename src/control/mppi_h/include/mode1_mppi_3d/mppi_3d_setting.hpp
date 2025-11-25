#pragma once

#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <grid_map_core/GridMap.hpp>
#include "common_type.hpp"
#include "mode1_mppi_3d/param.hpp"

namespace target_system_mppi_3d
{

// define 3D state space
using StateSpace3D = common_type::XYYaw; // check definition of XYYaw in common_type.hpp
static constexpr int DIM_STATE_SPACE = 3;
//// conversion function from XYYaw to StateSpace3D
inline common_type::XYYaw convertXYYawToStateSpace3D(const StateSpace3D& state)
{
    return state; // in mppi_3d, StateSpace3D is equal to XYYaw
}
//// conversion function from StateSpace3D to XYYaw
inline StateSpace3D convertStateSpace3DToXYYaw(const common_type::XYYaw& state)
{
    return state; // in mppi_3d, StateSpace3D is equal to XYYaw
}

// define 3D control input space
using ControlSpace3D = common_type::VxVyOmega; // check definition of VxVyOmega in common_type.hpp
static constexpr int DIM_CONTROL_SPACE = 3;
//// conversion function from VxVyOmega to ControlSpace3D
inline ControlSpace3D convertVxVyOmegaToControlSpace3D(const common_type::VxVyOmega& cmd)
{
    return cmd; // in mppi_3d, ControlSpace3D is equal to VxVyOmega
}
//// conversion function from ControlSpace3D to VxVyOmega
inline common_type::VxVyOmega convertControlSpace3DToVxVyOmega(const ControlSpace3D& cmd)
{
    return cmd; // in mppi_3d, ControlSpace3D is equal to VxVyOmega
}

// 8DoF vehicle command space
using ControlSpace8D = common_type::VehicleCommand8D; // check definition of VehicleCommand8D in common_type.hpp
static constexpr int DIM_VEHICLE_COMMAND_SPACE = 8;
//// conversion function from ControlSpace3D to ControlSpace8D
inline ControlSpace8D convertControlSpace3DToControlSpace8D(const ControlSpace3D& cmd, const param::MPPI3DParam& param)
{
    ControlSpace8D cmd8d;
    cmd8d.steer_fl = std::atan2(cmd.vy + param.target_system.l_f * cmd.omega, cmd.vx - param.target_system.d_l * cmd.omega);
    cmd8d.steer_fr = std::atan2(cmd.vy + param.target_system.l_f * cmd.omega, cmd.vx + param.target_system.d_r * cmd.omega);
    cmd8d.steer_rl = std::atan2(cmd.vy - param.target_system.l_r * cmd.omega, cmd.vx - param.target_system.d_l * cmd.omega);
    cmd8d.steer_rr = std::atan2(cmd.vy - param.target_system.l_r * cmd.omega, cmd.vx + param.target_system.d_r * cmd.omega);
    cmd8d.rotor_fl = std::sqrt(std::pow(cmd.vx - param.target_system.d_l * cmd.omega, 2) + std::pow(cmd.vy + param.target_system.l_f * cmd.omega, 2)) / param.target_system.tire_radius;
    cmd8d.rotor_fr = std::sqrt(std::pow(cmd.vx + param.target_system.d_r * cmd.omega, 2) + std::pow(cmd.vy + param.target_system.l_f * cmd.omega, 2)) / param.target_system.tire_radius;
    cmd8d.rotor_rl = std::sqrt(std::pow(cmd.vx - param.target_system.d_l * cmd.omega, 2) + std::pow(cmd.vy - param.target_system.l_r * cmd.omega, 2)) / param.target_system.tire_radius;
    cmd8d.rotor_rr = std::sqrt(std::pow(cmd.vx + param.target_system.d_r * cmd.omega, 2) + std::pow(cmd.vy - param.target_system.l_r * cmd.omega, 2)) / param.target_system.tire_radius;
    return cmd8d;
}

// define state updating rule
inline StateSpace3D calcNextState(const StateSpace3D& current_state, const ControlSpace3D& cmd, const double dt, double slip_factor = 0.0)
{
    // clamp control input
    ControlSpace3D clamped_cmd = cmd;
    clamped_cmd.clamp();

    // Apply Slip Model
    // v_y_actual = v_y_cmd - slip_factor * (v_x * omega)
    // This reduces the effective lateral velocity when cornering hard
    double vy_slip = - slip_factor * clamped_cmd.vx * clamped_cmd.omega;
    double vy_effective = clamped_cmd.vy + vy_slip;

    // calculate next state
    StateSpace3D next_state;
    next_state.x = current_state.x + clamped_cmd.vx * std::cos(current_state.yaw) * dt - vy_effective * std::sin(current_state.yaw) * dt;
    next_state.y = current_state.y + clamped_cmd.vx * std::sin(current_state.yaw) * dt + vy_effective * std::cos(current_state.yaw) * dt;
    next_state.yaw = current_state.yaw + clamped_cmd.omega * dt;
    next_state.unwrap(); // unwrap yaw angle

    // return next state
    return next_state;
}

} // namespace target_system_mppi_3d

namespace controller_mppi_3d
{

    // stage cost function
    inline double stage_cost(
        const target_system_mppi_3d::StateSpace3D& state,
        target_system_mppi_3d::ControlSpace3D& control_input,
        target_system_mppi_3d::ControlSpace3D& prev_control_input,
        const grid_map::GridMap& collision_costmap,
        const grid_map::GridMap& distance_error_map,
        const grid_map::GridMap& ref_yaw_map,
        const target_system_mppi_3d::StateSpace3D& goal_state,
        const param::MPPI3DParam& param,
        double slip_factor = 0.0
    )
    {
        // clamp control input
        control_input.clamp();
        prev_control_input.clamp();

        // initialize stage cost
        double cost = 0.0;

        // ========================================================================
        // [1] Slip Risk Penalty: Penalize control inputs that may cause lateral slip
        // Physical model: v_slip = slip_factor * v_x * omega (centrifugal-induced slip)
        // ========================================================================
        double slip_velocity = slip_factor * control_input.vx * control_input.omega;
        cost += param.controller.weight_slip_penalty * (slip_velocity * slip_velocity);

        // ========================================================================
        // [2] Curvature-Aware Adaptive Speed Regulation (for Omnidirectional Robots)
        // For swerve drive, we don't need to create turning radius like car-like robots.
        // Instead, we focus on limiting lateral acceleration to prevent slip.
        // ========================================================================
        if (ref_yaw_map.isInside(grid_map::Position(state.x, state.y)))
        {
            double ref_yaw_curr = ref_yaw_map.atPosition("ref_yaw", grid_map::Position(state.x, state.y), grid_map::InterpolationMethods::INTER_NEAREST);
            
            // Lookahead to estimate upcoming curvature
            double lookahead = param.controller.curvature_lookahead_dist;
            double next_x = state.x + lookahead * std::cos(ref_yaw_curr);
            double next_y = state.y + lookahead * std::sin(ref_yaw_curr);
            
            if (ref_yaw_map.isInside(grid_map::Position(next_x, next_y)))
            {
                double ref_yaw_ahead = ref_yaw_map.atPosition("ref_yaw", grid_map::Position(next_x, next_y), grid_map::InterpolationMethods::INTER_NEAREST);
                double delta_yaw = std::abs(std::remainder(ref_yaw_ahead - ref_yaw_curr, 2 * M_PI));
                double curvature = delta_yaw / lookahead; // Approximate path curvature [1/m]
                
                // Adaptive friction coefficient: higher slip_factor means lower effective friction
                double friction_degradation = 1.0 - 2.0 * slip_factor; // slip_factor in [0, 0.3]
                friction_degradation = std::max(0.3, friction_degradation);
                double mu_eff = param.controller.base_friction_coeff * friction_degradation;
                
                // For omnidirectional robot: use higher curvature floor since it can turn sharply
                const double g = 9.81;
                const double curvature_floor = 0.5; // Higher floor for omni robots
                double effective_curvature = std::max(curvature, curvature_floor);
                
                // Soft saturation for smooth transition
                double blend_factor = 1.0 / (1.0 + std::exp(-10.0 * (curvature - 0.3)));
                effective_curvature = curvature_floor + blend_factor * (curvature - curvature_floor);
                effective_curvature = std::max(effective_curvature, curvature_floor);
                
                double v_safe = std::sqrt(mu_eff * g / effective_curvature);
                v_safe = std::min(v_safe, param.controller.ref_velocity);
                
                // Only penalize if significantly exceeding safe speed
                double current_speed = std::sqrt(control_input.vx * control_input.vx + control_input.vy * control_input.vy);
                double speed_margin = 0.3; // 30% margin for omni robots
                if (current_speed > v_safe * (1.0 + speed_margin))
                {
                    double speed_excess = current_speed - v_safe * (1.0 + speed_margin);
                    cost += param.controller.weight_curvature_speed * speed_excess * speed_excess;
                }
                
                // ================================================================
                // [3] Yaw Rate Tracking for Omnidirectional Robots
                // Encourage rotating towards upcoming path direction
                // ================================================================
                double yaw_error_to_path = std::remainder(state.yaw - ref_yaw_ahead, 2 * M_PI);
                double omega_desired = -2.0 * yaw_error_to_path;
                omega_desired = std::max(-2.0, std::min(2.0, omega_desired));
                
                double omega_error = control_input.omega - omega_desired;
                cost += param.controller.weight_yaw_rate_error * omega_error * omega_error;
            }
        }

        // Track target velocity (considering only aligned component to the reference path)
        if (ref_yaw_map.isInside(grid_map::Position(state.x, state.y)))
        {
            double ref_yaw = ref_yaw_map.atPosition("ref_yaw", grid_map::Position(state.x, state.y), grid_map::InterpolationMethods::INTER_NEAREST);
            double diff_yaw = std::remainder(state.yaw - ref_yaw, 2 * M_PI); // diff_yaw is in [-pi, pi]
            Eigen::Matrix<double, 2, 1> ref_vel_direction;
            // for circular or square path tracking
            ref_vel_direction << std::cos(diff_yaw), -std::sin(diff_yaw);
            Eigen::Matrix<double, 2, 1> current_vel;
            current_vel << control_input.vx, control_input.vy;
            double ref_aligned_vel = ref_vel_direction.dot(current_vel);
            cost += param.controller.weight_velocity_error * pow(ref_aligned_vel - param.controller.ref_velocity, 2);
        }

        // Try to align with the reference path (heading error)
        if (ref_yaw_map.isInside(grid_map::Position(state.x, state.y)))
        {
            double ref_yaw = ref_yaw_map.atPosition("ref_yaw", grid_map::Position(state.x, state.y), grid_map::InterpolationMethods::INTER_NEAREST);
            double diff_yaw = std::remainder(state.yaw - ref_yaw, 2 * M_PI); // diff_yaw is in [-pi, pi]
            cost += param.controller.weight_angular_error * diff_yaw * diff_yaw;
        }

        // avoid collision
        if (collision_costmap.isInside(grid_map::Position(state.x, state.y)))
        {
            cost += param.controller.weight_collision_penalty * collision_costmap.atPosition("collision_cost", grid_map::Position(state.x, state.y));
        }

        // avoid large distance error from reference path
        if (distance_error_map.isInside(grid_map::Position(state.x, state.y)))
        {
            // change interpolation method more accurate
            double dist_error = distance_error_map.atPosition("distance_error", grid_map::Position(state.x, state.y), grid_map::InterpolationMethods::INTER_LINEAR);
            cost += param.controller.weight_distance_error_penalty * dist_error;
        }

        // penalize large control input change (vx, vy, omega)
        Eigen::Matrix<double, 1, target_system_mppi_3d::DIM_CONTROL_SPACE> weight_command_change(param.controller.weight_cmd_change.data());
        cost += weight_command_change * (control_input.eigen() - prev_control_input.eigen()).cwiseAbs2();

        // penalize large vehicle command change (steer_fl, steer_fr, steer_rl, steer_rr, rotor_fl, rotor_fr, rotor_rl, rotor_rr)
        target_system_mppi_3d::ControlSpace8D vehicle_command = target_system_mppi_3d::convertControlSpace3DToControlSpace8D(control_input, param);
        target_system_mppi_3d::ControlSpace8D prev_vehicle_command = target_system_mppi_3d::convertControlSpace3DToControlSpace8D(prev_control_input, param);
        Eigen::Matrix<double, 1, target_system_mppi_3d::DIM_VEHICLE_COMMAND_SPACE> weight_vehicle_command_change(param.controller.weight_vehicle_cmd_change.data());
        cost += weight_vehicle_command_change * (vehicle_command.eigen() - prev_vehicle_command.eigen()).cwiseAbs2();

        return cost;
    }

    // terminal cost function
    inline double terminal_cost(
        const target_system_mppi_3d::StateSpace3D& state,
        const target_system_mppi_3d::StateSpace3D& goal_state,
        const param::MPPI3DParam& param
    )
    {
        // initialize terminal cost
        double cost = 0.0;

        // only when the vehicle is not close to the goal
        // Note: this cost is needed to avoid the vehicle go far from the goal (i.e. avoid running in the opposite direction along the reference path)
        if( std::sqrt( pow(goal_state.x - state.x, 2) + pow(goal_state.y - state.y, 2) ) > param.navigation.xy_goal_tolerance )
        {
            // get closer to the goal
            cost = param.controller.weight_terminal_state_penalty * ( pow((state.x - goal_state.x), 2) + pow((state.y - goal_state.y), 2) );
        }

        return cost;
    }

} // namespace controller_mppi_3d

