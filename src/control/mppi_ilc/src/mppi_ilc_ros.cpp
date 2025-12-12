#include "mppi_ilc/mppi_ilc_ros.hpp"
#include <tf2/utils.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <limits>

namespace mppi_ilc
{

MPPIILCRos::MPPIILCRos(ros::NodeHandle& nh, ros::NodeHandle& private_nh)
    : nh_(nh), private_nh_(private_nh), tf_listener_(tf_buffer_)
{
    loadParameters();

    controller_ = std::make_unique<MPPIILCCore>(config_, ilc_cfg_);
    controller_->setFeedbackGains(feedback_gains_[0], feedback_gains_[1], feedback_gains_[2]);
    controller_->resizeILC(1);  // avoid empty bias

    // Subscribers
    odom_sub_ = nh_.subscribe("odom", 1, &MPPIILCRos::odomCallback, this);
    ref_path_sub_ = nh_.subscribe("reference_path", 1, &MPPIILCRos::refPathCallback, this);
    collision_costmap_sub_ = nh_.subscribe("collision_costmap", 1, &MPPIILCRos::collisionCostmapCallback, this);
    distance_error_map_sub_ = nh_.subscribe("distance_error_map", 1, &MPPIILCRos::distanceErrorMapCallback, this);
    ref_yaw_map_sub_ = nh_.subscribe("reference_yaw_map", 1, &MPPIILCRos::refYawMapCallback, this);

    // Publishers
    cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("cmd_vel", 1);
    optimal_traj_pub_ = nh_.advertise<nav_msgs::Path>("optimal_trajectory", 1);
    sampled_traj_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("sampled_trajectories", 1);
    calc_time_pub_ = nh_.advertise<std_msgs::Float32>("mppi_ilc_calc_time", 1);
    overlay_text_pub_ = nh_.advertise<jsk_rviz_plugins::OverlayText>("mppi_ilc_status", 1);
    eval_msg_pub_ = nh_.advertise<mppi_eval_msgs::MPPIEval>("mppi_ilc_eval", 1);

    double control_rate;
    private_nh_.param("control_rate", control_rate, 50.0);
    control_timer_ = nh_.createTimer(
        ros::Duration(1.0 / control_rate), &MPPIILCRos::controlTimerCallback, this);

    ROS_INFO("[MPPI-ILC] Controller initialized (ILC %s)", ilc_enabled_ ? "on" : "off");
}

void MPPIILCRos::loadParameters()
{
    // Vehicle
    private_nh_.param("vehicle/l_f", config_.vehicle.l_f, 0.5);
    private_nh_.param("vehicle/l_r", config_.vehicle.l_r, 0.5);
    private_nh_.param("vehicle/d_l", config_.vehicle.d_l, 0.5);
    private_nh_.param("vehicle/d_r", config_.vehicle.d_r, 0.5);
    private_nh_.param("vehicle/tire_radius", config_.vehicle.tire_radius, 0.2);
    private_nh_.param("vehicle/vx_max", config_.vehicle.vx_max, 3.0);
    private_nh_.param("vehicle/vy_max", config_.vehicle.vy_max, 3.0);
    private_nh_.param("vehicle/omega_max", config_.vehicle.omega_max, 3.0);
    private_nh_.param("vehicle/max_steer_angle", config_.vehicle.max_steer_angle, M_PI / 2.0);
    private_nh_.param("vehicle/max_wheel_vel", config_.vehicle.max_wheel_vel, 10.0);

    // MPPI
    private_nh_.param("mppi/num_samples", config_.mppi.num_samples, 2000);
    private_nh_.param("mppi/horizon", config_.mppi.prediction_horizon, 40);
    private_nh_.param("mppi/dt", config_.mppi.step_dt, 0.033);
    private_nh_.param("mppi/lambda", config_.mppi.lambda, 100.0);
    private_nh_.param("mppi/alpha", config_.mppi.alpha, 0.975);
    private_nh_.param("mppi/exploration_ratio", config_.mppi.exploration_ratio, 0.1);
    private_nh_.param("mppi/ref_velocity", config_.mppi.ref_velocity, 1.8);

    double noise_vx, noise_vy, noise_omega;
    private_nh_.param("mppi/noise_vx", noise_vx, 0.4);
    private_nh_.param("mppi/noise_vy", noise_vy, 0.4);
    private_nh_.param("mppi/noise_omega", noise_omega, 0.6);
    config_.mppi.sigma = Eigen::Vector3d(noise_vx, noise_vy, noise_omega);

    // Costs
    private_nh_.param("cost/distance_error", config_.weights.distance_error, 40.0);
    private_nh_.param("cost/angular_error", config_.weights.angular_error, 30.0);
    private_nh_.param("cost/velocity_error", config_.weights.velocity_error, 10.0);
    private_nh_.param("cost/terminal_state", config_.weights.terminal_state, 10.0);
    private_nh_.param("cost/collision", config_.weights.collision_penalty, 50.0);
    private_nh_.param("cost/slip_risk", config_.weights.slip_risk, 15.0);
    private_nh_.param("cost/curvature_speed", config_.weights.curvature_speed, 60.0);
    private_nh_.param("cost/yaw_rate_tracking", config_.weights.yaw_rate_tracking, 25.0);

    // Slip
    private_nh_.param("slip/learning_rate", config_.slip.learning_rate, 0.01);
    private_nh_.param("slip/slip_factor_min", config_.slip.slip_factor_min, 0.0);
    private_nh_.param("slip/slip_factor_max", config_.slip.slip_factor_max, 0.3);
    private_nh_.param("slip/excitation_threshold", config_.slip.excitation_threshold, 0.1);
    private_nh_.param("slip/base_friction_coeff", config_.slip.base_friction_coeff, 0.3);
    private_nh_.param("slip/curvature_lookahead", config_.slip.curvature_lookahead, 0.5);
    private_nh_.param("slip/curvature_floor", config_.slip.curvature_floor, 0.5);
    private_nh_.param("slip/speed_margin", config_.slip.speed_margin, 0.3);
    private_nh_.param("slip/enable_compensation", config_.slip.enable_compensation, true);
    private_nh_.param("slip/compensation_gain", config_.slip.compensation_gain, 0.7);

    // Feedback gains (for MPPI-HC closed-loop)
    double k_lateral, k_heading, k_integral;
    private_nh_.param("feedback/k_lateral", k_lateral, 2.5);
    private_nh_.param("feedback/k_heading", k_heading, 1.0);
    private_nh_.param("feedback/k_integral", k_integral, 0.5);
    config_.slip.enable_compensation = true;  // keep enabled for slip-aware core

    // Goal tolerance
    private_nh_.param("xy_goal_tolerance", config_.xy_goal_tolerance, 0.5);
    private_nh_.param("yaw_goal_tolerance", config_.yaw_goal_tolerance, 3.14);
    private_nh_.param("goal_proximity/threshold", goal_proximity_threshold_, 0.8);
    private_nh_.param("goal_proximity/feedback_fade", goal_feedback_fade_, true);

    // Closest-index continuity / self-intersection robustness
    private_nh_.param("tracking/idx_window_back", idx_window_back_, 25);
    private_nh_.param("tracking/idx_window_fwd", idx_window_fwd_, 60);
    private_nh_.param("tracking/idx_allow_wraparound", idx_allow_wraparound_, true);
    private_nh_.param("tracking/closed_path_threshold", idx_closed_path_threshold_, 0.6);
    private_nh_.param("tracking/idx_heading_weight", idx_heading_weight_, 0.4);
    private_nh_.param("tracking/idx_heading_gate", idx_heading_gate_, 1.2);
    private_nh_.param("tracking/global_fallback_factor", idx_global_fallback_factor_, 1.6);

    // ILC parameters
    private_nh_.param("ilc/enabled", ilc_enabled_, true);
    private_nh_.param("ilc/reset_on_new_path", reset_ilc_on_new_path_, true);
    private_nh_.param("ilc/k_lateral", ilc_cfg_.k_lateral, 0.15);
    private_nh_.param("ilc/k_heading", ilc_cfg_.k_heading, 0.05);
    private_nh_.param("ilc/decay", ilc_cfg_.decay, 0.995);
    private_nh_.param("ilc/max_bias_v", ilc_cfg_.max_bias_v, 0.6);
    private_nh_.param("ilc/max_bias_omega", ilc_cfg_.max_bias_omega, 0.8);
    private_nh_.param("ilc/curvature_threshold", ilc_cfg_.curvature_threshold, 0.10);
    private_nh_.param("ilc/error_deadband", ilc_cfg_.error_deadband, 0.005);
    private_nh_.param("ilc/max_update_lateral", ilc_cfg_.max_update_lateral, 0.02);
    private_nh_.param("ilc/max_update_heading", ilc_cfg_.max_update_heading, 0.02);
    ilc_cfg_.enabled = ilc_enabled_;

    // Set closed-loop gains inside slip compensator
    // NOTE: MPPI-HC exposes setter; we call later after controller creation
    feedback_gains_[0] = k_lateral;
    feedback_gains_[1] = k_heading;
    feedback_gains_[2] = k_integral;

    ROS_INFO("[MPPI-ILC] Parameters loaded");
}

void MPPIILCRos::odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
    current_state_.x = msg->pose.pose.position.x;
    current_state_.y = msg->pose.pose.position.y;
    current_state_.yaw = tf2::getYaw(msg->pose.pose.orientation);
    current_state_.vx = msg->twist.twist.linear.x;
    current_state_.vy = msg->twist.twist.linear.y;
    current_state_.omega = msg->twist.twist.angular.z;
    odom_received_ = true;

    // Keep slip estimator updated for the underlying MPPI-HC core
    controller_->updateEstimator(current_state_.vx, current_state_.vy, current_state_.omega);
}

void MPPIILCRos::refPathCallback(const nav_msgs::Path::ConstPtr& msg)
{
    if (msg->poses.empty()) {
        ROS_WARN_THROTTLE(1.0, "[MPPI-ILC] Received empty path");
        return;
    }

    ref_path_ = *msg;

    // Reset continuity state on new path message
    have_last_closest_idx_ = false;
    last_closest_idx_ = -1;

    // Determine if path is closed (for wrap-around indexing)
    path_is_closed_ = false;
    if (ref_path_.poses.size() >= 2) {
        const auto& p0 = ref_path_.poses.front().pose.position;
        const auto& pN = ref_path_.poses.back().pose.position;
        path_is_closed_ = (std::hypot(pN.x - p0.x, pN.y - p0.y) < idx_closed_path_threshold_);
    }
    const auto& goal_pose = msg->poses.back().pose;
    goal_state_.x = goal_pose.position.x;
    goal_state_.y = goal_pose.position.y;
    goal_state_.yaw = tf2::getYaw(goal_pose.orientation);
    goal_state_.vx = 0.0;
    goal_state_.vy = 0.0;
    goal_state_.omega = 0.0;

    path_received_ = true;
    if (reset_ilc_on_new_path_) {
        controller_->resizeILC(ref_path_.poses.size());
        controller_->resetILC();
        ROS_INFO("[MPPI-ILC] Reset ILC memory for new path (N=%zu)", ref_path_.poses.size());
    } else {
        controller_->resizeILC(ref_path_.poses.size());
    }
}

void MPPIILCRos::collisionCostmapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg)
{
    grid_map::GridMapRosConverter::fromOccupancyGrid(*msg, "collision_cost", collision_map_);
    costmaps_received_ = true;
}

void MPPIILCRos::distanceErrorMapCallback(const grid_map_msgs::GridMap::ConstPtr& msg)
{
    grid_map::GridMapRosConverter::fromMessage(*msg, distance_error_map_);
}

void MPPIILCRos::refYawMapCallback(const grid_map_msgs::GridMap::ConstPtr& msg)
{
    grid_map::GridMapRosConverter::fromMessage(*msg, ref_yaw_map_);
}

void MPPIILCRos::controlTimerCallback(const ros::TimerEvent& event)
{
    if (!odom_received_ || !path_received_ || !costmaps_received_) {
        ROS_WARN_THROTTLE(1.0, "[MPPI-ILC] Waiting for data... odom:%d path:%d maps:%d",
                          odom_received_, path_received_, costmaps_received_);
        return;
    }

    auto start_time = ros::Time::now();
    double dt = 0.02;
    if (!last_control_time_.isZero()) {
        dt = (start_time - last_control_time_).toSec();
        dt = std::clamp(dt, 0.001, 0.1);
    }
    last_control_time_ = start_time;

    double dist_to_goal = std::hypot(goal_state_.x - current_state_.x,
                                     goal_state_.y - current_state_.y);

    TrackingError error = computeTrackingError();

    // Update continuity state for next cycle
    last_closest_idx_ = error.closest_idx;
    have_last_closest_idx_ = true;

    // Low-speed guard
    double current_speed = std::hypot(current_state_.vx, current_state_.vy);
    const double low_speed_threshold = 0.3;
    if (current_speed < low_speed_threshold) {
        double speed_factor = current_speed / low_speed_threshold;
        error.lateral_error *= speed_factor;
        error.heading_error *= speed_factor;
    }

    // Spin guard
    double linear_vel = std::hypot(current_state_.vx, current_state_.vy);
    double angular_ratio = std::abs(current_state_.omega) / (linear_vel + 0.01);
    if (angular_ratio > 2.0) {
        double spin_factor = std::min(1.0, 2.0 / angular_ratio);
        error.lateral_error *= spin_factor;
    }

    // Large error guard
    const double large_error_threshold = 0.5;
    if (std::abs(error.lateral_error) > large_error_threshold) {
        double factor = large_error_threshold / std::abs(error.lateral_error);
        error.lateral_error *= factor;
    }

    // Goal fade-out
    if (goal_feedback_fade_ && dist_to_goal < goal_proximity_threshold_) {
        double fade = dist_to_goal / goal_proximity_threshold_;
        fade = std::max(0.0, fade);
        error.lateral_error *= fade;
        error.heading_error *= fade;
        error.path_curvature *= fade;
    }

    TrackingContext ctx;
    ctx.lateral_error = error.lateral_error;
    ctx.heading_error = error.heading_error;
    ctx.path_curvature = error.path_curvature;
    ctx.closest_idx = error.closest_idx;

    mppi_hc::BodyVelocity cmd = controller_->solveWithILC(
        current_state_,
        collision_map_,
        distance_error_map_,
        ref_yaw_map_,
        goal_state_,
        ctx,
        dt);

    // Online ILC update
    if (ilc_enabled_ && path_received_) {
        // Gating to reduce noisy updates in sharp turns and tiny errors
        double lat_err = error.lateral_error;
        double head_err = error.heading_error;

        // Deadband
        if (std::abs(lat_err) < ilc_cfg_.error_deadband) lat_err = 0.0;
        if (std::abs(head_err) < ilc_cfg_.error_deadband) head_err = 0.0;

        // Reduce learning on tight curvature
        if (std::abs(error.path_curvature) > ilc_cfg_.curvature_threshold) {
            lat_err *= 0.3;  // down-weight lateral learning on turns
            head_err *= 0.3;
        }

        // Limit per-step contribution to avoid sudden bias jumps
        lat_err = std::clamp(lat_err, -ilc_cfg_.max_update_lateral, ilc_cfg_.max_update_lateral);
        head_err = std::clamp(head_err, -ilc_cfg_.max_update_heading, ilc_cfg_.max_update_heading);

        controller_->updateILC(error.closest_idx, lat_err, head_err);
    }

    publishCommand(cmd);
    publishVisualization();
    publishEvalMessage();

    auto calc_time = (ros::Time::now() - start_time).toSec() * 1000.0;
    std_msgs::Float32 time_msg;
    time_msg.data = calc_time;
    calc_time_pub_.publish(time_msg);

    ROS_INFO_THROTTLE(2.0, "[MPPI-ILC] Lat_err: %.3f m, Head_err: %.1f deg, curv: %.2f, idx:%d",
                      error.lateral_error, error.heading_error * 180.0 / M_PI,
                      error.path_curvature, error.closest_idx);
}

void MPPIILCRos::publishCommand(const mppi_hc::BodyVelocity& cmd)
{
    geometry_msgs::Twist twist;
    twist.linear.x = cmd.vx;
    twist.linear.y = cmd.vy;
    twist.angular.z = cmd.omega;
    cmd_vel_pub_.publish(twist);
}

void MPPIILCRos::publishVisualization()
{
    const auto opt_traj = controller_->getOptimalTrajectory();
    nav_msgs::Path path_msg;
    path_msg.header.stamp = ros::Time::now();
    path_msg.header.frame_id = "odom";

    for (const auto& state : opt_traj) {
        geometry_msgs::PoseStamped pose;
        pose.header = path_msg.header;
        pose.pose.position.x = state.x;
        pose.pose.position.y = state.y;
        pose.pose.position.z = 0.0;

        tf2::Quaternion q;
        q.setRPY(0, 0, state.yaw);
        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();
        path_msg.poses.push_back(pose);
    }
    optimal_traj_pub_.publish(path_msg);

    const auto stats = controller_->getEstimatorStats();
    jsk_rviz_plugins::OverlayText overlay;
    overlay.action = jsk_rviz_plugins::OverlayText::ADD;
    overlay.width = 320;
    overlay.height = 170;
    overlay.left = 10;
    overlay.top = 10;
    overlay.bg_color.r = 0.0;
    overlay.bg_color.g = 0.0;
    overlay.bg_color.b = 0.0;
    overlay.bg_color.a = 0.7;
    overlay.fg_color.r = 0.2;
    overlay.fg_color.g = 1.0;
    overlay.fg_color.b = 0.2;
    overlay.fg_color.a = 1.0;
    overlay.text_size = 12;
    overlay.font = "DejaVu Sans Mono";

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "=== MPPI-ILC ===" << std::endl;
    ss << "ILC: " << (ilc_enabled_ ? "on" : "off") << std::endl;
    ss << "K_slip:     " << stats.current_k_slip << std::endl;
    ss << "Error:      " << stats.estimation_error << std::endl;
    ss << "Samples:    " << stats.num_samples << std::endl;
    overlay.text = ss.str();
    overlay_text_pub_.publish(overlay);
}

void MPPIILCRos::publishEvalMessage()
{
    mppi_eval_msgs::MPPIEval eval;
    eval.header.stamp = ros::Time::now();
    eval.header.frame_id = "odom";

    eval.global_x = current_state_.x;
    eval.global_y = current_state_.y;
    eval.global_yaw = current_state_.yaw;
    eval.cmd_vx = current_state_.vx;
    eval.cmd_vy = current_state_.vy;
    eval.cmd_yawrate = current_state_.omega;
    eval.state_cost = controller_->getEstimatorStats().current_k_slip;

    double pos_error = std::hypot(goal_state_.x - current_state_.x,
                                  goal_state_.y - current_state_.y);
    eval.goal_reached = (pos_error < 0.1);
    eval_msg_pub_.publish(eval);
}

MPPIILCRos::TrackingError MPPIILCRos::computeTrackingError() const
{
    TrackingError error;
    if (ref_path_.poses.size() < 2) {
        return error;
    }

    const int n = static_cast<int>(ref_path_.poses.size());

    auto wrapAngle = [](double a) {
        while (a > M_PI) a -= 2.0 * M_PI;
        while (a < -M_PI) a += 2.0 * M_PI;
        return a;
    };

    auto distToIdx = [&](int i) {
        const auto& p = ref_path_.poses[static_cast<std::size_t>(i)].pose.position;
        return std::hypot(p.x - current_state_.x, p.y - current_state_.y);
    };

    auto headingErrAtIdx = [&](int i) {
        const double path_yaw = tf2::getYaw(ref_path_.poses[static_cast<std::size_t>(i)].pose.orientation);
        return wrapAngle(current_state_.yaw - path_yaw);
    };

    // Global nearest (distance only) for fallback/recovery
    int global_idx = 0;
    double global_dist = std::numeric_limits<double>::max();
    for (int i = 0; i < n; ++i) {
        const double d = distToIdx(i);
        if (d < global_dist) {
            global_dist = d;
            global_idx = i;
        }
    }

    int best_idx = global_idx;
    double best_dist = global_dist;
    double best_head = headingErrAtIdx(best_idx);

    if (have_last_closest_idx_ && last_closest_idx_ >= 0 && last_closest_idx_ < n) {
        const int back = std::max(0, idx_window_back_);
        const int fwd = std::max(0, idx_window_fwd_);

        auto evalCandidate = [&](int i, int& out_idx, double& out_score, double& out_dist, double& out_head) {
            const double d = distToIdx(i);
            const double h = headingErrAtIdx(i);
            const double score = d + idx_heading_weight_ * std::abs(h);
            if (score < out_score) {
                out_score = score;
                out_idx = i;
                out_dist = d;
                out_head = h;
            }
        };

        int win_best_idx = last_closest_idx_;
        double win_best_score = std::numeric_limits<double>::max();
        double win_best_dist = std::numeric_limits<double>::max();
        double win_best_head = 0.0;

        int i_min = std::max(0, last_closest_idx_ - back);
        int i_max = std::min(n - 1, last_closest_idx_ + fwd);
        for (int i = i_min; i <= i_max; ++i) {
            evalCandidate(i, win_best_idx, win_best_score, win_best_dist, win_best_head);
        }

        if (path_is_closed_ && idx_allow_wraparound_) {
            if (last_closest_idx_ + fwd >= n) {
                const int wrap_max = (last_closest_idx_ + fwd) - (n - 1);
                for (int i = 0; i <= std::min(n - 1, wrap_max); ++i) {
                    evalCandidate(i, win_best_idx, win_best_score, win_best_dist, win_best_head);
                }
            }
            if (last_closest_idx_ - back < 0) {
                const int wrap_min = n + (last_closest_idx_ - back);
                for (int i = std::max(0, wrap_min); i < n; ++i) {
                    evalCandidate(i, win_best_idx, win_best_score, win_best_dist, win_best_head);
                }
            }
        }

        const bool heading_bad = (std::abs(win_best_head) > idx_heading_gate_);
        const bool dist_bad = (win_best_dist > idx_global_fallback_factor_ * std::max(1e-6, global_dist));
        if (!heading_bad && !dist_bad) {
            best_idx = win_best_idx;
            best_dist = win_best_dist;
            best_head = win_best_head;
        } else {
            best_idx = global_idx;
            best_dist = global_dist;
            best_head = headingErrAtIdx(best_idx);
        }
    }

    error.closest_idx = best_idx;

    double end_factor = 1.0;
    if (best_idx > n - 5) {
        end_factor = static_cast<double>(n - best_idx) / 5.0;
        end_factor = std::max(0.0, end_factor);
    }

    double path_yaw = tf2::getYaw(ref_path_.poses[best_idx].pose.orientation);
    double dx = current_state_.x - ref_path_.poses[best_idx].pose.position.x;
    double dy = current_state_.y - ref_path_.poses[best_idx].pose.position.y;
    error.lateral_error = (-dx * std::sin(path_yaw) + dy * std::cos(path_yaw)) * end_factor;

    error.heading_error = wrapAngle(current_state_.yaw - path_yaw) * end_factor;

    int i0 = std::max(0, best_idx - 3);
    int i2 = std::min(n - 1, best_idx + 3);
    if (i2 > i0 + 1) {
        double x0 = ref_path_.poses[i0].pose.position.x;
        double y0 = ref_path_.poses[i0].pose.position.y;
        double x1 = ref_path_.poses[best_idx].pose.position.x;
        double y1 = ref_path_.poses[best_idx].pose.position.y;
        double x2 = ref_path_.poses[i2].pose.position.x;
        double y2 = ref_path_.poses[i2].pose.position.y;
        double area2 = std::abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
        double d01 = std::hypot(x1 - x0, y1 - y0);
        double d12 = std::hypot(x2 - x1, y2 - y1);
        double d20 = std::hypot(x0 - x2, y0 - y2);
        double denom = d01 * d12 * d20;
        if (denom > 1e-6) {
            error.path_curvature = area2 / denom;
            double cross = (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1);
            error.path_curvature = std::copysign(error.path_curvature, cross);
        }
    }
    error.path_curvature *= end_factor;
    return error;
}

}  // namespace mppi_ilc
