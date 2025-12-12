#pragma once

#include <vector>
#include <algorithm>
#include <cstddef>
#include "mppi_hc/types.hpp"

namespace mppi_ilc_prior
{

using BodyVelocity = mppi_hc::BodyVelocity;

struct ILCPriorLearningConfig
{
    bool enabled = true;

    // Learning: map tracking error -> feedforward bias
    double k_lateral = 0.15;      // vy bias per lateral error
    double k_heading = 0.05;      // omega bias per heading error
    double decay = 0.995;         // bias decay

    // Clamp
    double max_bias_v = 0.6;
    double max_bias_omega = 0.8;

    // Update gating (handled at ROS layer)
    double curvature_threshold = 0.10;
    double error_deadband = 0.005;
    double max_update_lateral = 0.02;
    double max_update_heading = 0.02;

    // Prior injection
    double prior_weight = 0.0;            // quadratic regularization weight
    bool prior_apply_to_exploration = true;
    int prior_index_step = 1;             // idx(t) = closest_idx + t * step
};

class ILCMemory
{
public:
    void configure(const ILCPriorLearningConfig& cfg) { config_ = cfg; }

    void resize(std::size_t n) { bias_.assign(n, BodyVelocity{}); }

    void reset() { std::fill(bias_.begin(), bias_.end(), BodyVelocity{}); }

    std::size_t size() const { return bias_.size(); }

    struct BiasStats
    {
        double rms_vx = 0.0;
        double rms_vy = 0.0;
        double rms_omega = 0.0;
        double max_abs_vx = 0.0;
        double max_abs_vy = 0.0;
        double max_abs_omega = 0.0;
    };

    BiasStats computeStats() const
    {
        BiasStats s;
        if (bias_.empty()) return s;

        double sum_vx2 = 0.0;
        double sum_vy2 = 0.0;
        double sum_w2 = 0.0;
        for (const auto& b : bias_) {
            sum_vx2 += b.vx * b.vx;
            sum_vy2 += b.vy * b.vy;
            sum_w2 += b.omega * b.omega;
            s.max_abs_vx = std::max(s.max_abs_vx, std::abs(b.vx));
            s.max_abs_vy = std::max(s.max_abs_vy, std::abs(b.vy));
            s.max_abs_omega = std::max(s.max_abs_omega, std::abs(b.omega));
        }
        const double n = static_cast<double>(bias_.size());
        s.rms_vx = std::sqrt(sum_vx2 / n);
        s.rms_vy = std::sqrt(sum_vy2 / n);
        s.rms_omega = std::sqrt(sum_w2 / n);
        return s;
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

        bias_[idx].vx *= config_.decay;
        bias_[idx].vy *= config_.decay;
        bias_[idx].omega *= config_.decay;

        bias_[idx].vy += config_.k_lateral * lateral_error;
        bias_[idx].omega += config_.k_heading * heading_error;

        bias_[idx].vx = std::clamp(bias_[idx].vx, -config_.max_bias_v, config_.max_bias_v);
        bias_[idx].vy = std::clamp(bias_[idx].vy, -config_.max_bias_v, config_.max_bias_v);
        bias_[idx].omega = std::clamp(bias_[idx].omega, -config_.max_bias_omega, config_.max_bias_omega);
    }

private:
    ILCPriorLearningConfig config_{};
    std::vector<BodyVelocity> bias_;
};

}  // namespace mppi_ilc_prior
