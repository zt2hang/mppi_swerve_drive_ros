#pragma once

#include <vector>
#include <algorithm>
#include <cstddef>
#include "mppi_hc/types.hpp"

namespace mppi_ilc
{

using BodyVelocity = mppi_hc::BodyVelocity;

struct ILCLearningConfig
{
    bool enabled = true;
    double k_lateral = 0.15;      // gain for lateral error -> vy bias
    double k_heading = 0.05;      // gain for heading error -> omega bias
    double decay = 0.995;         // bias decay per cycle to avoid wind-up
    double max_bias_v = 0.6;      // clamp for vx/vy feedforward [m/s]
    double max_bias_omega = 0.8;  // clamp for omega feedforward [rad/s]
    double curvature_threshold = 0.10; // above this reduce learning
    double error_deadband = 0.005;     // ignore tiny errors
    double max_update_lateral = 0.02;  // cap per-step update from lateral error
    double max_update_heading = 0.02;  // cap per-step update from heading error
};

class ILCMemory
{
public:
    ILCMemory() = default;

    void configure(const ILCLearningConfig& cfg)
    {
        config_ = cfg;
    }

    void resize(std::size_t n)
    {
        bias_.assign(n, BodyVelocity{});
    }

    void reset()
    {
        std::fill(bias_.begin(), bias_.end(), BodyVelocity{});
    }

    BodyVelocity getBias(std::size_t idx) const
    {
        if (bias_.empty()) return BodyVelocity{};
        idx = std::min(idx, bias_.size() - 1);
        return bias_[idx];
    }

    void update(std::size_t idx, double lateral_error, double heading_error)
    {
        if (!config_.enabled || bias_.empty()) return;
        idx = std::min(idx, bias_.size() - 1);

        // Decay old bias to prevent divergence
        bias_[idx].vx *= config_.decay;
        bias_[idx].vy *= config_.decay;
        bias_[idx].omega *= config_.decay;

        // Apply proportional learning
        bias_[idx].vy += config_.k_lateral * lateral_error;
        bias_[idx].omega += config_.k_heading * heading_error;

        // Clamp to keep feedforward physically plausible
        bias_[idx].vx = std::clamp(bias_[idx].vx, -config_.max_bias_v, config_.max_bias_v);
        bias_[idx].vy = std::clamp(bias_[idx].vy, -config_.max_bias_v, config_.max_bias_v);
        bias_[idx].omega = std::clamp(bias_[idx].omega, -config_.max_bias_omega, config_.max_bias_omega);
    }

private:
    ILCLearningConfig config_{};
    std::vector<BodyVelocity> bias_;
};

}  // namespace mppi_ilc
