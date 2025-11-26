/**
 * @file cost_function.cpp
 * @brief Implementation of MPPI-HC cost function
 */

#include "mppi_hc/cost_function.hpp"
#include <cmath>

namespace mppi_hc
{

CostFunction::CostFunction(const CostWeights& weights, const SlipParams& slip_params,
                           const VehicleParams& vehicle_params)
    : weights_(weights)
    , slip_params_(slip_params)
    , vehicle_params_(vehicle_params)
    , dynamics_(vehicle_params)
{
}

double CostFunction::stageCost(
    const State& state,
    const Control& control,
    const Control& prev_control,
    const grid_map::GridMap& collision_map,
    const grid_map::GridMap& distance_error_map,
    const grid_map::GridMap& ref_yaw_map,
    const State& goal,
    double slip_factor
) const
{
    double cost = 0.0;
    grid_map::Position pos(state.x, state.y);

    // ========================================================================
    // [1] Slip Risk Penalty
    // ========================================================================
    cost += slipRiskCost(control, slip_factor);

    // ========================================================================
    // [2] Curvature-Aware Speed Regulation (for Omnidirectional Robots)
    // ========================================================================
    cost += curvatureSpeedCost(state, control, ref_yaw_map, slip_factor);

    // ========================================================================
    // [3] Path Tracking Costs
    // ========================================================================
    cost += pathTrackingCost(state, control, distance_error_map, ref_yaw_map);

    // ========================================================================
    // [4] Collision Avoidance
    // ========================================================================
    if (collision_map.isInside(pos)) {
        double collision_cost = collision_map.atPosition("collision_cost", pos);
        cost += weights_.collision_penalty * collision_cost;
    }

    // ========================================================================
    // [5] Control Smoothness
    // ========================================================================
    Eigen::Vector3d du = control.toEigen() - prev_control.toEigen();
    for (int i = 0; i < 3; ++i) {
        cost += weights_.cmd_change[i] * du(i) * du(i);
    }

    // Vehicle-level command smoothness
    VehicleCommand8D cmd = dynamics_.bodyToWheelCommands(control);
    VehicleCommand8D prev_cmd = dynamics_.bodyToWheelCommands(prev_control);
    Eigen::Matrix<double, 8, 1> dv = cmd.toEigen() - prev_cmd.toEigen();
    for (int i = 0; i < 8; ++i) {
        cost += weights_.vehicle_cmd_change[i] * dv(i) * dv(i);
    }

    return cost;
}

double CostFunction::terminalCost(const State& state, const State& goal) const
{
    double dx = state.x - goal.x;
    double dy = state.y - goal.y;
    double dist_sq = dx * dx + dy * dy;

    // Only apply terminal cost if not near goal
    if (dist_sq > 0.25) {  // > 0.5m from goal
        return weights_.terminal_state * dist_sq;
    }
    return 0.0;
}

double CostFunction::slipRiskCost(const Control& control, double slip_factor) const
{
    // v_slip = K_slip * v_x * omega
    double slip_velocity = slip_factor * control.vx * control.omega;
    return weights_.slip_risk * slip_velocity * slip_velocity;
}

double CostFunction::curvatureSpeedCost(
    const State& state,
    const Control& control,
    const grid_map::GridMap& ref_yaw_map,
    double slip_factor
) const
{
    double cost = 0.0;
    grid_map::Position pos(state.x, state.y);

    if (!ref_yaw_map.isInside(pos)) {
        return 0.0;
    }

    double ref_yaw_curr = ref_yaw_map.atPosition("ref_yaw", pos, 
                          grid_map::InterpolationMethods::INTER_NEAREST);

    // Lookahead for curvature estimation
    double lookahead = slip_params_.curvature_lookahead;
    double next_x = state.x + lookahead * std::cos(ref_yaw_curr);
    double next_y = state.y + lookahead * std::sin(ref_yaw_curr);
    grid_map::Position next_pos(next_x, next_y);

    if (!ref_yaw_map.isInside(next_pos)) {
        return 0.0;
    }

    double ref_yaw_ahead = ref_yaw_map.atPosition("ref_yaw", next_pos,
                           grid_map::InterpolationMethods::INTER_NEAREST);
    double delta_yaw = std::abs(std::remainder(ref_yaw_ahead - ref_yaw_curr, 2.0 * M_PI));
    double curvature = delta_yaw / lookahead;

    // Adaptive friction coefficient
    double friction_degradation = 1.0 - 2.0 * slip_factor;
    friction_degradation = std::max(0.3, friction_degradation);
    double mu_eff = slip_params_.base_friction_coeff * friction_degradation;

    // For omnidirectional robot: use higher curvature floor
    const double g = 9.81;
    double curvature_floor = slip_params_.curvature_floor;
    
    // Soft saturation using sigmoid blend
    double blend = 1.0 / (1.0 + std::exp(-10.0 * (curvature - 0.3)));
    double eff_curvature = curvature_floor + blend * (curvature - curvature_floor);
    eff_curvature = std::max(eff_curvature, curvature_floor);

    // Safe cornering speed
    double v_safe = std::sqrt(mu_eff * g / eff_curvature);
    double current_speed = control.speed();

    // Penalize only if significantly exceeding safe speed
    double threshold = v_safe * (1.0 + slip_params_.speed_margin);
    if (current_speed > threshold) {
        double excess = current_speed - threshold;
        cost += weights_.curvature_speed * excess * excess;
    }

    // Yaw rate tracking for omnidirectional robots
    // Only activate during turns (when curvature is significant)
    // On straight lines, let the heading error cost handle alignment
    const double curvature_threshold = 0.2;  // Only track yaw rate when turning
    if (curvature > curvature_threshold) {
        // Encourage rotating towards upcoming path direction
        double yaw_error = std::remainder(state.yaw - ref_yaw_ahead, 2.0 * M_PI);
        double omega_desired = -2.0 * yaw_error;  // P-controller style
        omega_desired = std::clamp(omega_desired, -2.0, 2.0);
        
        // Scale weight by curvature intensity
        double curvature_scale = std::min(1.0, (curvature - curvature_threshold) / 0.5);
        double omega_error = control.omega - omega_desired;
        cost += weights_.yaw_rate_tracking * curvature_scale * omega_error * omega_error;
    }

    return cost;
}

double CostFunction::pathTrackingCost(
    const State& state,
    const Control& control,
    const grid_map::GridMap& distance_error_map,
    const grid_map::GridMap& ref_yaw_map
) const
{
    double cost = 0.0;
    grid_map::Position pos(state.x, state.y);

    // Distance error (primary tracking objective)
    if (distance_error_map.isInside(pos)) {
        double dist_error = distance_error_map.atPosition("distance_error", pos,
                            grid_map::InterpolationMethods::INTER_LINEAR);
        cost += weights_.distance_error * dist_error;
    }

    // Angular error and velocity tracking
    if (ref_yaw_map.isInside(pos)) {
        double ref_yaw = ref_yaw_map.atPosition("ref_yaw", pos,
                         grid_map::InterpolationMethods::INTER_LINEAR);
        double diff_yaw = std::remainder(state.yaw - ref_yaw, 2.0 * M_PI);
        
        // Heading error - use smooth cost (Huber-like for large errors)
        double abs_yaw_err = std::abs(diff_yaw);
        if (abs_yaw_err < 0.5) {
            // Quadratic for small errors
            cost += weights_.angular_error * diff_yaw * diff_yaw;
        } else {
            // Linear for large errors (prevents extreme costs)
            cost += weights_.angular_error * (abs_yaw_err - 0.25);
        }

        // Velocity tracking (aligned with reference path)
        double ref_velocity = 2.0;  // TODO: make configurable
        double aligned_vel = control.vx * std::cos(diff_yaw) + control.vy * std::sin(diff_yaw);
        cost += weights_.velocity_error * std::pow(aligned_vel - ref_velocity, 2);
    }

    return cost;
}

double CostFunction::computeSafeSpeed(double curvature, double mu_eff) const
{
    const double g = 9.81;
    double safe_curvature = std::max(curvature, slip_params_.curvature_floor);
    return std::sqrt(mu_eff * g / safe_curvature);
}

double CostFunction::computeDesiredYawRate(const State& state, double ref_yaw_ahead) const
{
    double yaw_error = std::remainder(state.yaw - ref_yaw_ahead, 2.0 * M_PI);
    double omega_desired = -2.0 * yaw_error;
    return std::clamp(omega_desired, -2.0, 2.0);
}

} // namespace mppi_hc
