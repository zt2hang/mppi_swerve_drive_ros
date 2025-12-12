#include "mppi_ilc_prior/mppi_ilc_prior_core.hpp"

#include <algorithm>

namespace mppi_ilc_prior
{

MPPIILCPriorCore::MPPIILCPriorCore(const mppi_hc::ControllerConfig& config,
                                   const ILCPriorLearningConfig& ilc_cfg)
    : config_(config), ilc_cfg_(ilc_cfg), mppi_core_(config)
{
    ilc_memory_.configure(ilc_cfg_);
}

void MPPIILCPriorCore::applyILCPriorFromIndices(const std::vector<int>& indices)
{
    const int T = config_.mppi.prediction_horizon;
    if (!ilc_cfg_.enabled) {
        mppi_core_.clearControlPrior();
        return;
    }
    if (static_cast<int>(indices.size()) != T || T <= 0) {
        // Ignore mismatch to avoid out-of-bounds.
        return;
    }

    mppi_hc::ControlSequence prior_seq;
    prior_seq.resize(T);
    for (int t = 0; t < T; ++t) {
        const int idx = std::max(0, indices[t]);
        prior_seq[t] = ilc_memory_.getBias(static_cast<std::size_t>(idx));
    }
    mppi_core_.setControlPrior(prior_seq, ilc_cfg_.prior_weight, ilc_cfg_.prior_apply_to_exploration);
}

void MPPIILCPriorCore::clearMPPIControlPrior()
{
    mppi_core_.clearControlPrior();
}

mppi_hc::BodyVelocity MPPIILCPriorCore::solveWithILCPrior(
    const mppi_hc::State& current_state,
    const grid_map::GridMap& collision_map,
    const grid_map::GridMap& distance_error_map,
    const grid_map::GridMap& ref_yaw_map,
    const mppi_hc::State& goal,
    const TrackingContext& error_ctx,
    double dt)
{
    return mppi_core_.solveWithFeedback(
        current_state,
        collision_map,
        distance_error_map,
        ref_yaw_map,
        goal,
        error_ctx.lateral_error,
        error_ctx.heading_error,
        error_ctx.path_curvature,
        dt);
}

void MPPIILCPriorCore::setConfig(const mppi_hc::ControllerConfig& config)
{
    config_ = config;
    mppi_core_.setConfig(config_);
}

void MPPIILCPriorCore::updateILC(int path_idx, double lateral_error, double heading_error)
{
    ilc_memory_.update(static_cast<std::size_t>(std::max(0, path_idx)), lateral_error, heading_error);
}

void MPPIILCPriorCore::resizeILC(std::size_t n_points)
{
    ilc_memory_.resize(n_points);
}

void MPPIILCPriorCore::resetILC()
{
    ilc_memory_.reset();
}

}  // namespace mppi_ilc_prior
