/**
 * @file cost_function.cpp
 * @brief Implementation of tire force-aware cost function
 * 
 * @author ZZT
 * @date 2024
 */

#include "mppi_tf/cost_function.hpp"
#include <cmath>
#include <algorithm>

namespace mppi_tf
{

CostFunction::CostFunction(const CostWeights& weights, const VehicleParams& vehicle)
    : weights_(weights)
    , vehicle_(vehicle)
    , ref_velocity_(2.0)
{
}

// ============================================================================
// Main Cost Computation
// ============================================================================

double CostFunction::stageCostForce(
    const FullState& state,
    const BodyForce& force,
    const BodyForce& prev_force,
    const WheelForces& wheel_forces,
    const grid_map::GridMap& collision_map,
    const grid_map::GridMap& distance_map,
    const grid_map::GridMap& yaw_map,
    const FullState& goal) const
{
    double cost = 0.0;
    
    // Path tracking
    cost += pathTrackingCost(state, distance_map, yaw_map);
    
    // Velocity tracking
    cost += velocityTrackingCost(state, state.velocity(), yaw_map);
    
    // Collision avoidance
    cost += collisionCost(state, collision_map);
    
    // Tire force costs (key novelty)
    cost += frictionUtilizationCost(wheel_forces);
    cost += frictionMarginCost(wheel_forces);
    
    // Force smoothness
    cost += forceRateCost(force, prev_force);
    
    // Curvature-aware speed
    cost += curvatureSpeedCost(state, yaw_map);
    
    // Lateral acceleration
    cost += lateralAccelCost(state);
    
    return cost;
}

double CostFunction::stageCostVelocity(
    const FullState& state,
    const BodyVelocity& vel,
    const BodyVelocity& prev_vel,
    const grid_map::GridMap& collision_map,
    const grid_map::GridMap& distance_map,
    const grid_map::GridMap& yaw_map,
    const FullState& goal,
    double slip_factor) const
{
    double cost = 0.0;
    
    // Path tracking
    cost += pathTrackingCost(state, distance_map, yaw_map);
    
    // Velocity tracking
    cost += velocityTrackingCost(state, vel, yaw_map);
    
    // Collision avoidance
    cost += collisionCost(state, collision_map);
    
    // Velocity smoothness
    cost += velocityRateCost(vel, prev_vel);
    
    // Curvature-aware speed
    cost += curvatureSpeedCost(state, yaw_map);
    
    // Slip risk cost (similar to MPPI-HC)
    // Higher slip factor + high vx*omega = high slip risk
    double slip_risk = slip_factor * std::abs(vel.vx * vel.omega);
    cost += weights_.slip_angle * slip_risk * slip_risk;
    
    return cost;
}

double CostFunction::terminalCost(const FullState& state, const FullState& goal) const
{
    return goalCost(state, goal);
}

// ============================================================================
// Path Tracking Costs
// ============================================================================

double CostFunction::pathTrackingCost(
    const FullState& state,
    const grid_map::GridMap& distance_map,
    const grid_map::GridMap& yaw_map) const
{
    double cost = 0.0;
    
    // Distance to path
    if (isInsideMap(state.x, state.y, distance_map)) {
        try {
            double dist_error = distance_map.atPosition(
                "distance_error", 
                grid_map::Position(state.x, state.y),
                grid_map::InterpolationMethods::INTER_LINEAR
            );
            cost += weights_.distance_error * dist_error * dist_error;
        } catch (...) {
            // Map query failed, skip this cost
        }
    }
    
    // Heading alignment
    if (isInsideMap(state.x, state.y, yaw_map)) {
        try {
            double ref_yaw = yaw_map.atPosition(
                "ref_yaw",
                grid_map::Position(state.x, state.y),
                grid_map::InterpolationMethods::INTER_NEAREST
            );
            double yaw_error = std::remainder(state.yaw - ref_yaw, 2.0 * M_PI);
            cost += weights_.angular_error * yaw_error * yaw_error;
        } catch (...) {
            // Map query failed
        }
    }
    
    return cost;
}

double CostFunction::velocityTrackingCost(
    const FullState& state,
    const BodyVelocity& vel,
    const grid_map::GridMap& yaw_map) const
{
    double cost = 0.0;
    
    // Get reference direction (path tangent)
    if (isInsideMap(state.x, state.y, yaw_map)) {
        try {
            double ref_yaw = yaw_map.atPosition(
                "ref_yaw",
                grid_map::Position(state.x, state.y),
                grid_map::InterpolationMethods::INTER_NEAREST
            );
            
            // Angle between robot heading and path
            double diff_yaw = std::remainder(state.yaw - ref_yaw, 2.0 * M_PI);
            
            // Project velocity onto reference direction
            Eigen::Vector2d ref_dir(std::cos(diff_yaw), std::sin(diff_yaw));
            Eigen::Vector2d current_vel(vel.vx, vel.vy);
            double aligned_vel = ref_dir.dot(current_vel);
            
            // Penalize deviation from reference velocity
            double vel_error = aligned_vel - ref_velocity_;
            cost += weights_.velocity_error * vel_error * vel_error;
        } catch (...) {
            // Fallback: just penalize speed deviation
            double speed = vel.speed();
            double vel_error = speed - ref_velocity_;
            cost += weights_.velocity_error * vel_error * vel_error;
        }
    }
    
    return cost;
}

double CostFunction::collisionCost(
    const FullState& state,
    const grid_map::GridMap& collision_map) const
{
    if (!isInsideMap(state.x, state.y, collision_map)) {
        return weights_.collision_penalty;  // Outside map = collision
    }
    
    try {
        double collision_cost = collision_map.atPosition(
            "collision_cost",
            grid_map::Position(state.x, state.y)
        );
        return weights_.collision_penalty * collision_cost;
    } catch (...) {
        return 0.0;
    }
}

// ============================================================================
// Tire Force Costs (Novel)
// ============================================================================

double CostFunction::frictionUtilizationCost(const WheelForces& wheel_forces) const
{
    double total_cost = 0.0;
    
    for (int i = 0; i < NUM_WHEELS; ++i) {
        WheelIndex idx = static_cast<WheelIndex>(i);
        double Fz = vehicle_.mass.getStaticWheelLoad(idx);
        double F_max = mu_ * Fz;
        
        if (F_max < EPSILON) continue;
        
        double F_mag = wheel_forces[i].magnitude();
        double utilization = F_mag / F_max;
        
        // Quadratic penalty on utilization
        // Cost increases sharply as utilization approaches 1
        total_cost += utilization * utilization;
        
        // Extra penalty if exceeding friction limit
        if (utilization > 1.0) {
            total_cost += 10.0 * (utilization - 1.0) * (utilization - 1.0);
        }
    }
    
    return weights_.force_utilization * total_cost;
}

double CostFunction::frictionMarginCost(const WheelForces& wheel_forces) const
{
    double min_margin = 1.0;
    
    for (int i = 0; i < NUM_WHEELS; ++i) {
        WheelIndex idx = static_cast<WheelIndex>(i);
        double Fz = vehicle_.mass.getStaticWheelLoad(idx);
        double F_max = mu_ * Fz;
        
        if (F_max < EPSILON) continue;
        
        double F_mag = wheel_forces[i].magnitude();
        double margin = 1.0 - F_mag / F_max;
        min_margin = std::min(min_margin, margin);
    }
    
    // Penalize low margin (reward high margin)
    // Cost is high when margin is low
    double margin_deficit = std::max(0.0, 0.2 - min_margin);  // Target: 20% margin
    return weights_.friction_margin * margin_deficit * margin_deficit;
}

double CostFunction::slipAngleCost(
    const FullState& state,
    const std::array<double, NUM_WHEELS>& slip_angles) const
{
    double cost = 0.0;
    
    // Penalize large slip angles (nonlinear tire region)
    const double alpha_linear_limit = 0.1;  // ~6 degrees
    
    for (int i = 0; i < NUM_WHEELS; ++i) {
        double alpha = std::abs(slip_angles[i]);
        
        if (alpha > alpha_linear_limit) {
            // Quadratic penalty for exceeding linear region
            double excess = alpha - alpha_linear_limit;
            cost += excess * excess;
        }
    }
    
    return weights_.slip_angle * cost;
}

// ============================================================================
// Smoothness Costs
// ============================================================================

double CostFunction::forceRateCost(const BodyForce& force, const BodyForce& prev_force) const
{
    double dFx = force.Fx - prev_force.Fx;
    double dFy = force.Fy - prev_force.Fy;
    double dMz = force.Mz - prev_force.Mz;
    
    // Normalize by force limits for scale-invariance
    double Fx_max = vehicle_.getMaxLongForce();
    double Fy_max = vehicle_.getMaxLatForce();
    double Mz_max = vehicle_.getMaxYawMoment();
    
    double cost = (dFx * dFx) / (Fx_max * Fx_max + EPSILON)
                + (dFy * dFy) / (Fy_max * Fy_max + EPSILON)
                + (dMz * dMz) / (Mz_max * Mz_max + EPSILON);
    
    return weights_.force_change * cost;
}

double CostFunction::velocityRateCost(const BodyVelocity& vel, const BodyVelocity& prev_vel) const
{
    double dvx = vel.vx - prev_vel.vx;
    double dvy = vel.vy - prev_vel.vy;
    double domega = vel.omega - prev_vel.omega;
    
    // Use weights from config
    double cost = weights_.wheel_cmd_change[0] * dvx * dvx
                + weights_.wheel_cmd_change[1] * dvy * dvy
                + weights_.wheel_cmd_change[2] * domega * domega;
    
    return cost;
}

// ============================================================================
// Curvature and Acceleration Costs
// ============================================================================

double CostFunction::curvatureSpeedCost(
    const FullState& state,
    const grid_map::GridMap& yaw_map) const
{
    // Estimate local curvature
    double curvature = estimateCurvature(state, yaw_map);
    
    // For omnidirectional robots, use a floor on effective curvature
    curvature = std::max(curvature, 0.5);  // Min curvature radius = 2m
    
    // Compute safe speed
    double v_safe = computeSafeSpeed(curvature);
    
    // Current speed
    double speed = state.speed();
    
    // Penalize exceeding safe speed
    if (speed > v_safe) {
        double excess = speed - v_safe;
        return weights_.curvature_speed * excess * excess;
    }
    
    return 0.0;
}

double CostFunction::lateralAccelCost(const FullState& state) const
{
    // Lateral acceleration = vx * omega (body frame)
    double ay_approx = state.vx * state.omega;
    
    // Maximum comfortable lateral acceleration
    double ay_max = mu_ * GRAVITY * 0.8;  // 80% of friction limit
    
    if (std::abs(ay_approx) > ay_max) {
        double excess = std::abs(ay_approx) - ay_max;
        return weights_.curvature_speed * 0.5 * excess * excess;
    }
    
    return 0.0;
}

double CostFunction::goalCost(const FullState& state, const FullState& goal) const
{
    double dx = state.x - goal.x;
    double dy = state.y - goal.y;
    double dist_sq = dx * dx + dy * dy;
    
    return weights_.terminal_state * dist_sq;
}

// ============================================================================
// Helper Functions
// ============================================================================

double CostFunction::computeSafeSpeed(double curvature) const
{
    // v = sqrt(mu * g / kappa) for circular motion
    // Add safety margin
    if (curvature < EPSILON) {
        return vehicle_.limits.vx_max;
    }
    
    double v_limit = std::sqrt(mu_ * GRAVITY / curvature);
    
    // Apply margin
    v_limit *= 0.8;
    
    // Clamp to vehicle limits
    return std::min(v_limit, vehicle_.limits.vx_max);
}

double CostFunction::estimateCurvature(
    const FullState& state,
    const grid_map::GridMap& yaw_map) const
{
    // Estimate curvature from yaw gradient
    if (!isInsideMap(state.x, state.y, yaw_map)) {
        return 0.5;  // Default curvature
    }
    
    try {
        // Sample yaw at current position and ahead
        double ref_yaw_here = yaw_map.atPosition(
            "ref_yaw",
            grid_map::Position(state.x, state.y),
            grid_map::InterpolationMethods::INTER_NEAREST
        );
        
        // Look ahead in current direction
        double lookahead = 0.5;  // meters
        double x_ahead = state.x + lookahead * std::cos(state.yaw);
        double y_ahead = state.y + lookahead * std::sin(state.yaw);
        
        if (!isInsideMap(x_ahead, y_ahead, yaw_map)) {
            return 0.5;
        }
        
        double ref_yaw_ahead = yaw_map.atPosition(
            "ref_yaw",
            grid_map::Position(x_ahead, y_ahead),
            grid_map::InterpolationMethods::INTER_NEAREST
        );
        
        // Curvature ≈ dθ/ds
        double dyaw = std::remainder(ref_yaw_ahead - ref_yaw_here, 2.0 * M_PI);
        double curvature = std::abs(dyaw) / lookahead;
        
        return curvature;
    } catch (...) {
        return 0.5;
    }
}

bool CostFunction::isInsideMap(double x, double y, const grid_map::GridMap& map) const
{
    return map.isInside(grid_map::Position(x, y));
}

} // namespace mppi_tf
