#pragma once

#include "mppi_hc/mppi_hc_core.hpp"
#include "mppi_ilc_prior/ilc_memory.hpp"
#include <grid_map_core/GridMap.hpp>
#include <vector>

namespace mppi_ilc_prior
{

struct TrackingContext
{
    double lateral_error = 0.0;
    double heading_error = 0.0;
    double path_curvature = 0.0;
    int closest_idx = 0;
};

class MPPIILCPriorCore
{
public:
    MPPIILCPriorCore(const mppi_hc::ControllerConfig& config,
                     const ILCPriorLearningConfig& ilc_cfg);

    mppi_hc::BodyVelocity solveWithILCPrior(
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

    mppi_hc::BodyVelocity getILCBiasAt(int path_idx) const
    {
        return ilc_memory_.getBias(static_cast<std::size_t>(std::max(0, path_idx)));
    }

    ILCMemory::BiasStats getILCBiasStats() const { return ilc_memory_.computeStats(); }

    // Apply MPPI control prior from ILC memory using an externally provided
    // index sequence (size=T). Each entry selects which path point bias to use.
    void applyILCPriorFromIndices(const std::vector<int>& indices);
    void clearMPPIControlPrior();

    mppi_hc::StateSequence getOptimalTrajectory() const { return mppi_core_.getOptimalTrajectory(); }
    mppi_hc::VehicleCommand8D getWheelCommands() const { return mppi_core_.getWheelCommands(); }
    mppi_hc::SlipEstimator::Statistics getEstimatorStats() const { return mppi_core_.getEstimatorStats(); }

private:
    mppi_hc::ControllerConfig config_;
    ILCPriorLearningConfig ilc_cfg_;
    ILCMemory ilc_memory_;
    mppi_hc::MPPIHCCore mppi_core_;
};

}  // namespace mppi_ilc_prior
