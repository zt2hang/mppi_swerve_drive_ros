#include "mppi_ilc/mppi_ilc_core.hpp"

namespace mppi_ilc
{

MPPIILCCore::MPPIILCCore(const mppi_hc::ControllerConfig& config,
                         const ILCLearningConfig& ilc_cfg)
    : config_(config), ilc_cfg_(ilc_cfg), mppi_core_(config)
{
    ilc_memory_.configure(ilc_cfg_);
}

mppi_hc::BodyVelocity MPPIILCCore::solveWithILC(
    const mppi_hc::State& current_state,
    const grid_map::GridMap& collision_map,
    const grid_map::GridMap& distance_error_map,
    const grid_map::GridMap& ref_yaw_map,
    const mppi_hc::State& goal,
    const TrackingContext& error_ctx,
    double dt)
{
    // First, run the base MPPI-HC controller with feedback
    mppi_hc::BodyVelocity cmd = mppi_core_.solveWithFeedback(
        current_state,
        collision_map,
        distance_error_map,
        ref_yaw_map,
        goal,
        error_ctx.lateral_error,
        error_ctx.heading_error,
        error_ctx.path_curvature,
        dt);

    // Apply learned feedforward for this path index
    if (ilc_cfg_.enabled) {
        const auto bias = ilc_memory_.getBias(static_cast<std::size_t>(error_ctx.closest_idx));
        cmd.vx += bias.vx;
        cmd.vy += bias.vy;
        cmd.omega += bias.omega;
        cmd.clamp(config_.vehicle.vx_max, config_.vehicle.vy_max, config_.vehicle.omega_max);
    }

    return cmd;
}

void MPPIILCCore::setConfig(const mppi_hc::ControllerConfig& config)
{
    config_ = config;
    mppi_core_.setConfig(config_);
}

void MPPIILCCore::updateILC(int path_idx, double lateral_error, double heading_error)
{
    ilc_memory_.update(static_cast<std::size_t>(std::max(0, path_idx)), lateral_error, heading_error);
}

void MPPIILCCore::resizeILC(std::size_t n_points)
{
    ilc_memory_.resize(n_points);
}

void MPPIILCCore::resetILC()
{
    ilc_memory_.reset();
}

}  // namespace mppi_ilc
