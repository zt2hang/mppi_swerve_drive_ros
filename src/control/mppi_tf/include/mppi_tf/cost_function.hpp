#pragma once

/**
 * @file cost_function.hpp
 * @brief Tire force-aware cost function for MPPI
 * 
 * This module defines cost functions that incorporate tire force considerations:
 * 
 * 1. Path Tracking Costs
 *    - Distance to reference path
 *    - Heading alignment
 *    - Velocity tracking
 * 
 * 2. Safety Costs
 *    - Collision avoidance
 *    - Friction margin preservation
 *    - Rollover prevention (lateral acceleration limit)
 * 
 * 3. Tire/Force-Related Costs (Novel contributions)
 *    - Friction utilization penalty (efficiency)
 *    - Force rate-of-change penalty (smoothness)
 *    - Slip angle regularization (stability)
 *    - Force allocation fairness (wear distribution)
 * 
 * 4. Curvature-Aware Costs
 *    - Speed regulation based on path curvature
 *    - Anticipatory deceleration for upcoming curves
 * 
 * The cost function supports both:
 * - Force-space control (Fx, Fy, Mz)
 * - Velocity-space control (vx, vy, omega)
 * 
 * @author ZZT
 * @date 2024
 */

#include "mppi_tf/types.hpp"
#include "mppi_tf/dynamics.hpp"
#include <grid_map_core/GridMap.hpp>

namespace mppi_tf
{

/**
 * @brief Tire force-aware cost function for MPPI
 */
class CostFunction
{
public:
    /**
     * @brief Construct cost function
     * @param weights Cost weights
     * @param vehicle Vehicle parameters
     */
    CostFunction(const CostWeights& weights, const VehicleParams& vehicle);

    /**
     * @brief Update cost weights
     */
    void setWeights(const CostWeights& weights) { weights_ = weights; }
    const CostWeights& getWeights() const { return weights_; }

    /**
     * @brief Set current friction estimate (affects force costs)
     */
    void setFrictionCoeff(double mu) { mu_ = std::clamp(mu, 0.1, 1.5); }

    // ========================================================================
    // Main Cost Computation
    // ========================================================================

    /**
     * @brief Compute total stage cost (force-space version)
     * 
     * @param state Current state
     * @param force Applied body force
     * @param prev_force Previous force (for rate penalty)
     * @param wheel_forces Per-wheel forces (for utilization cost)
     * @param collision_map Collision cost grid
     * @param distance_map Distance to reference path grid
     * @param yaw_map Reference heading grid
     * @param goal Goal state
     * @return Total stage cost
     */
    double stageCostForce(
        const FullState& state,
        const BodyForce& force,
        const BodyForce& prev_force,
        const WheelForces& wheel_forces,
        const grid_map::GridMap& collision_map,
        const grid_map::GridMap& distance_map,
        const grid_map::GridMap& yaw_map,
        const FullState& goal
    ) const;

    /**
     * @brief Compute total stage cost (velocity-space version)
     * 
     * For compatibility with velocity-based MPPI.
     */
    double stageCostVelocity(
        const FullState& state,
        const BodyVelocity& vel,
        const BodyVelocity& prev_vel,
        const grid_map::GridMap& collision_map,
        const grid_map::GridMap& distance_map,
        const grid_map::GridMap& yaw_map,
        const FullState& goal,
        double slip_factor
    ) const;

    /**
     * @brief Compute terminal cost
     */
    double terminalCost(const FullState& state, const FullState& goal) const;

    // ========================================================================
    // Individual Cost Components (for analysis/debugging)
    // ========================================================================

    /**
     * @brief Path tracking cost
     * 
     * Penalizes deviation from reference path.
     */
    double pathTrackingCost(
        const FullState& state,
        const grid_map::GridMap& distance_map,
        const grid_map::GridMap& yaw_map
    ) const;

    /**
     * @brief Velocity tracking cost
     * 
     * Penalizes deviation from reference velocity.
     */
    double velocityTrackingCost(
        const FullState& state,
        const BodyVelocity& vel,
        const grid_map::GridMap& yaw_map
    ) const;

    /**
     * @brief Collision avoidance cost
     */
    double collisionCost(
        const FullState& state,
        const grid_map::GridMap& collision_map
    ) const;

    /**
     * @brief Friction utilization cost
     * 
     * Penalizes high friction utilization to maintain safety margin.
     * Cost increases as wheels approach friction limit.
     */
    double frictionUtilizationCost(const WheelForces& wheel_forces) const;

    /**
     * @brief Friction margin reward
     * 
     * Rewards maintaining friction margin (inverse of utilization).
     * Helps keep tires in stable operating region.
     */
    double frictionMarginCost(const WheelForces& wheel_forces) const;

    /**
     * @brief Slip angle cost
     * 
     * Penalizes large slip angles that lead to nonlinear tire behavior.
     */
    double slipAngleCost(
        const FullState& state,
        const std::array<double, NUM_WHEELS>& slip_angles
    ) const;

    /**
     * @brief Force rate-of-change cost
     * 
     * Penalizes rapid force changes for smooth control.
     */
    double forceRateCost(const BodyForce& force, const BodyForce& prev_force) const;

    /**
     * @brief Velocity rate-of-change cost
     */
    double velocityRateCost(const BodyVelocity& vel, const BodyVelocity& prev_vel) const;

    /**
     * @brief Curvature-aware speed cost
     * 
     * Penalizes exceeding safe speed for current/upcoming path curvature.
     */
    double curvatureSpeedCost(
        const FullState& state,
        const grid_map::GridMap& yaw_map
    ) const;

    /**
     * @brief Lateral acceleration cost
     * 
     * Penalizes exceeding comfortable/safe lateral acceleration.
     */
    double lateralAccelCost(const FullState& state) const;

    /**
     * @brief Goal proximity cost
     * 
     * Attracts trajectory towards goal.
     */
    double goalCost(const FullState& state, const FullState& goal) const;

private:
    CostWeights weights_;
    VehicleParams vehicle_;
    double mu_ = 0.9;  // Current friction estimate
    
    // Reference velocity (can be updated)
    double ref_velocity_ = 2.0;

    /**
     * @brief Compute safe speed for given curvature
     * 
     * v_safe = sqrt(mu * g / kappa)
     */
    double computeSafeSpeed(double curvature) const;

    /**
     * @brief Estimate local path curvature from yaw map
     */
    double estimateCurvature(
        const FullState& state,
        const grid_map::GridMap& yaw_map
    ) const;

    /**
     * @brief Check if position is inside map bounds
     */
    bool isInsideMap(double x, double y, const grid_map::GridMap& map) const;
};

} // namespace mppi_tf
