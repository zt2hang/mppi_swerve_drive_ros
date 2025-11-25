#pragma once

#include <string>
#include <vector>

namespace param
{

struct MPPI4DParam
{
    struct Navigation
    {
        double xy_goal_tolerance;
        double yaw_goal_tolerance;
    };
    Navigation navigation;

    struct TargetSystem
    {
        double l_f;
        double l_r;
        double d_l;
        double d_r;
        double tire_radius;
    };
    TargetSystem target_system;

    struct Controller
    {
        std::string name;
        double control_interval;
        int num_samples;
        int prediction_horizon;
        double step_len_sec;
        double param_exploration;
        double param_lambda;
        double param_alpha;
        std::vector<double> sigma;
        bool reduce_computation;
        std::vector<double> weight_cmd_change;
        std::vector<double> weight_vehicle_cmd_change;
        double ref_velocity;
        double weight_velocity_error;
        double weight_angular_error;
        double weight_collision_penalty;
        double weight_distance_error_penalty;
        double weight_terminal_state_penalty;
        double weight_slip_penalty; // New parameter for slip cost
        // Curvature-aware adaptive speed regulation parameters
        double curvature_lookahead_dist;     // Lookahead distance for curvature estimation [m]
        double base_friction_coeff;          // Nominal friction coefficient (e.g., 0.8 for dry, 0.3 for wet)
        double weight_curvature_speed;       // Weight for penalizing speed exceeding safe curvature limit
        double weight_yaw_rate_error;        // Weight for yaw rate tracking error
        bool use_sg_filter;
        int sg_filter_half_window_size;
        int sg_filter_poly_order;
        bool use_adaptive_estimator;
    };
    Controller controller;
};

} // namespace param
