#pragma once

#include "mppi_hc/mppi_hc_core.hpp"
#include "mppi_ilc/ilc_memory.hpp"
#include <grid_map_core/GridMap.hpp>

namespace mppi_ilc
{

struct TrackingContext
{
    double lateral_error = 0.0;
    double heading_error = 0.0;
    double path_curvature = 0.0;
    int closest_idx = 0;
};

class MPPIILCCore
{
public:
    MPPIILCCore(const mppi_hc::ControllerConfig& config,
                const ILCLearningConfig& ilc_cfg);

    mppi_hc::BodyVelocity solveWithILC(
        const mppi_hc::State& current_state,
        const grid_map::GridMap& collision_map,
        const grid_map::GridMap& distance_error_map,
        const grid_map::GridMap& ref_yaw_map,
        const mppi_hc::State& goal,
        const TrackingContext& error_ctx,
        double dt);

    void setConfig(const mppi_hc::ControllerConfig& config);
    void setFeedbackGains(double k_lateral, double k_heading, double k_integral)
    {
        mppi_core_.setFeedbackGains(k_lateral, k_heading, k_integral);
    }
    void updateEstimator(double actual_vx, double actual_vy, double actual_omega)
    {
        mppi_core_.updateEstimator(actual_vx, actual_vy, actual_omega);
    }
    void resetCompensatorIntegrator() { mppi_core_.resetCompensatorIntegrator(); }
    const mppi_hc::ControllerConfig& getConfig() const { return config_; }

    void updateILC(int path_idx, double lateral_error, double heading_error);
    void resizeILC(std::size_t n_points);
    void resetILC();

    const mppi_hc::StateSequence& getOptimalTrajectory() const { return mppi_core_.getOptimalTrajectory(); }
    mppi_hc::VehicleCommand8D getWheelCommands() const { return mppi_core_.getWheelCommands(); }
    mppi_hc::SlipEstimator::Statistics getEstimatorStats() const { return mppi_core_.getEstimatorStats(); }

private:
    mppi_hc::ControllerConfig config_;
    ILCLearningConfig ilc_cfg_;
    ILCMemory ilc_memory_;
    mppi_hc::MPPIHCCore mppi_core_;
};

}  // namespace mppi_ilc
