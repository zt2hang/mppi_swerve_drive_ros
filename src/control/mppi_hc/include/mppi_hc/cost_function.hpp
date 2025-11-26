#pragma once

/**
 * @file cost_function.hpp
 * @brief Cost function components for MPPI-HC
 * 
 * Includes:
 *   - Path tracking costs (distance, heading, velocity)
 *   - Safety costs (collision avoidance)
 *   - Slip-aware costs (slip risk, curvature speed, yaw rate)
 *   - Smoothness costs (control change)
 */

#include "mppi_hc/types.hpp"
#include "mppi_hc/dynamics.hpp"
#include <grid_map_core/GridMap.hpp>

namespace mppi_hc
{

/**
 * @brief Cost function calculator for MPPI
 */
class CostFunction
{
public:
    CostFunction(const CostWeights& weights, const SlipParams& slip_params,
                 const VehicleParams& vehicle_params);

    /**
     * @brief Compute total stage cost
     */
    double stageCost(
        const State& state,
        const Control& control,
        const Control& prev_control,
        const grid_map::GridMap& collision_map,
        const grid_map::GridMap& distance_error_map,
        const grid_map::GridMap& ref_yaw_map,
        const State& goal,
        double slip_factor
    ) const;

    /**
     * @brief Compute terminal cost
     */
    double terminalCost(const State& state, const State& goal) const;

    // Individual cost components (for debugging/analysis)
    double slipRiskCost(const Control& control, double slip_factor) const;
    double curvatureSpeedCost(const State& state, const Control& control,
                              const grid_map::GridMap& ref_yaw_map, double slip_factor) const;
    double pathTrackingCost(const State& state, const Control& control,
                            const grid_map::GridMap& distance_error_map,
                            const grid_map::GridMap& ref_yaw_map) const;

    void setWeights(const CostWeights& weights) { weights_ = weights; }
    const CostWeights& getWeights() const { return weights_; }

private:
    CostWeights weights_;
    SlipParams slip_params_;
    VehicleParams vehicle_params_;
    DynamicsModel dynamics_;

    // Helper methods
    double computeSafeSpeed(double curvature, double mu_eff) const;
    double computeDesiredYawRate(const State& state, double ref_yaw_ahead) const;
};

} // namespace mppi_hc
