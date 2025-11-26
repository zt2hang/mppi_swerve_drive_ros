/**
 * @file mppi_hc_ros.cpp
 * @brief ROS wrapper implementation for MPPI-HC controller
 */

#include "mppi_hc/mppi_hc_ros.hpp"
#include <tf2/utils.h>

namespace mppi_hc
{

MPPIHCRos::MPPIHCRos(ros::NodeHandle& nh, ros::NodeHandle& private_nh)
    : nh_(nh), private_nh_(private_nh), tf_listener_(tf_buffer_)
{
    // Load parameters
    loadParameters();

    // Create controller
    controller_ = std::make_unique<MPPIHCCore>(config_);
    
    // Set feedback gains for closed-loop precision tracking
    controller_->setFeedbackGains(feedback_gains_[0], feedback_gains_[1], feedback_gains_[2]);
    ROS_INFO("[MPPI-HC] Feedback gains set: k_lat=%.2f, k_head=%.2f, k_int=%.2f",
             feedback_gains_[0], feedback_gains_[1], feedback_gains_[2]);

    // Subscribers
    odom_sub_ = nh_.subscribe("odom", 1, &MPPIHCRos::odomCallback, this);
    ref_path_sub_ = nh_.subscribe("reference_path", 1, &MPPIHCRos::refPathCallback, this);
    collision_costmap_sub_ = nh_.subscribe(
        "collision_costmap", 1, &MPPIHCRos::collisionCostmapCallback, this);
    distance_error_map_sub_ = nh_.subscribe(
        "distance_error_map", 1, &MPPIHCRos::distanceErrorMapCallback, this);
    ref_yaw_map_sub_ = nh_.subscribe(
        "reference_yaw_map", 1, &MPPIHCRos::refYawMapCallback, this);

    // Publishers
    cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("cmd_vel", 1);
    optimal_traj_pub_ = nh_.advertise<nav_msgs::Path>("optimal_trajectory", 1);
    sampled_traj_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("sampled_trajectories", 1);
    calc_time_pub_ = nh_.advertise<std_msgs::Float32>("mppi_calc_time", 1);
    overlay_text_pub_ = nh_.advertise<jsk_rviz_plugins::OverlayText>("mppi_hc_status", 1);
    eval_msg_pub_ = nh_.advertise<mppi_eval_msgs::MPPIEval>("mppi_eval", 1);

    // Control timer
    double control_rate;
    private_nh_.param("control_rate", control_rate, 50.0);
    control_timer_ = nh_.createTimer(
        ros::Duration(1.0 / control_rate), &MPPIHCRos::controlTimerCallback, this);

    ROS_INFO("[MPPI-HC] Controller initialized successfully");
}

void MPPIHCRos::loadParameters()
{
    // Vehicle parameters
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

    // MPPI parameters
    private_nh_.param("mppi/num_samples", config_.mppi.num_samples, 3000);
    private_nh_.param("mppi/horizon", config_.mppi.prediction_horizon, 50);
    private_nh_.param("mppi/dt", config_.mppi.step_dt, 0.033);
    private_nh_.param("mppi/lambda", config_.mppi.lambda, 100.0);
    private_nh_.param("mppi/alpha", config_.mppi.alpha, 0.975);
    private_nh_.param("mppi/exploration_ratio", config_.mppi.exploration_ratio, 0.1);
    private_nh_.param("mppi/ref_velocity", config_.mppi.ref_velocity, 2.0);
    
    // Noise parameters
    double noise_vx, noise_vy, noise_omega;
    private_nh_.param("mppi/noise_vx", noise_vx, 0.5);
    private_nh_.param("mppi/noise_vy", noise_vy, 0.5);
    private_nh_.param("mppi/noise_omega", noise_omega, 0.8);
    config_.mppi.sigma = Eigen::Vector3d(noise_vx, noise_vy, noise_omega);

    // Cost weights
    private_nh_.param("cost/distance_error", config_.weights.distance_error, 40.0);
    private_nh_.param("cost/angular_error", config_.weights.angular_error, 30.0);
    private_nh_.param("cost/velocity_error", config_.weights.velocity_error, 10.0);
    private_nh_.param("cost/terminal_state", config_.weights.terminal_state, 10.0);
    private_nh_.param("cost/collision", config_.weights.collision_penalty, 50.0);
    private_nh_.param("cost/slip_risk", config_.weights.slip_risk, 15.0);
    private_nh_.param("cost/curvature_speed", config_.weights.curvature_speed, 60.0);
    private_nh_.param("cost/yaw_rate_tracking", config_.weights.yaw_rate_tracking, 25.0);

    // Slip parameters
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

    // Feedback gains for closed-loop precision tracking
    double k_lateral, k_heading, k_integral;
    private_nh_.param("feedback/k_lateral", k_lateral, 2.5);
    private_nh_.param("feedback/k_heading", k_heading, 1.0);
    private_nh_.param("feedback/k_integral", k_integral, 0.5);
    
    // Store feedback gains in slip params (will be set after controller creation)
    feedback_gains_[0] = k_lateral;
    feedback_gains_[1] = k_heading;
    feedback_gains_[2] = k_integral;

    ROS_INFO("[MPPI-HC] Parameters loaded");
}

void MPPIHCRos::odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
    // Extract pose
    current_state_.x = msg->pose.pose.position.x;
    current_state_.y = msg->pose.pose.position.y;
    current_state_.yaw = tf2::getYaw(msg->pose.pose.orientation);

    // Extract velocity
    current_state_.vx = msg->twist.twist.linear.x;
    current_state_.vy = msg->twist.twist.linear.y;
    current_state_.omega = msg->twist.twist.angular.z;

    odom_received_ = true;

    // Update estimator with actual velocities
    controller_->updateEstimator(
        current_state_.vx, current_state_.vy, current_state_.omega);
}

void MPPIHCRos::refPathCallback(const nav_msgs::Path::ConstPtr& msg)
{
    if (msg->poses.empty()) {
        ROS_WARN_THROTTLE(1.0, "[MPPI-HC] Received empty path");
        return;
    }

    // Store full path for error calculation
    ref_path_ = *msg;

    // Get goal from last pose
    const auto& goal_pose = msg->poses.back().pose;
    goal_state_.x = goal_pose.position.x;
    goal_state_.y = goal_pose.position.y;
    goal_state_.yaw = tf2::getYaw(goal_pose.orientation);
    goal_state_.vx = 0.0;
    goal_state_.vy = 0.0;
    goal_state_.omega = 0.0;

    path_received_ = true;
}

void MPPIHCRos::collisionCostmapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg)
{
    // Convert OccupancyGrid to GridMap (same as mppi_h)
    grid_map::GridMapRosConverter::fromOccupancyGrid(*msg, "collision_cost", collision_map_);
    costmaps_received_ = true;
}

void MPPIHCRos::distanceErrorMapCallback(const grid_map_msgs::GridMap::ConstPtr& msg)
{
    grid_map::GridMapRosConverter::fromMessage(*msg, distance_error_map_);
}

void MPPIHCRos::refYawMapCallback(const grid_map_msgs::GridMap::ConstPtr& msg)
{
    grid_map::GridMapRosConverter::fromMessage(*msg, ref_yaw_map_);
}

void MPPIHCRos::controlTimerCallback(const ros::TimerEvent& event)
{
    if (!odom_received_ || !path_received_ || !costmaps_received_) {
        ROS_WARN_THROTTLE(1.0, "[MPPI-HC] Waiting for data... odom:%d path:%d maps:%d",
                         odom_received_, path_received_, costmaps_received_);
        return;
    }

    auto start_time = ros::Time::now();
    
    // Compute time delta
    double dt = 0.02;  // default
    if (!last_control_time_.isZero()) {
        dt = (start_time - last_control_time_).toSec();
        dt = std::clamp(dt, 0.001, 0.1);  // Sanity check
    }
    last_control_time_ = start_time;

    // Compute tracking errors for closed-loop compensation
    TrackingError error = computeTrackingError();

    // Solve MPPI with closed-loop feedback compensation
    BodyVelocity cmd = controller_->solveWithFeedback(
        current_state_, 
        collision_map_, 
        distance_error_map_, 
        ref_yaw_map_, 
        goal_state_,
        error.lateral_error,
        error.heading_error,
        error.path_curvature,
        dt
    );

    // Publish command
    publishCommand(cmd);

    // Publish visualization
    publishVisualization();

    // Publish eval message
    publishEvalMessage();

    // Publish calculation time
    auto calc_time = (ros::Time::now() - start_time).toSec() * 1000.0;  // ms
    std_msgs::Float32 time_msg;
    time_msg.data = calc_time;
    calc_time_pub_.publish(time_msg);
    
    // Log tracking performance periodically
    ROS_INFO_THROTTLE(2.0, "[MPPI-HC] Lat_err: %.3fm, Head_err: %.1fdeg, Curv: %.2f",
                     error.lateral_error, error.heading_error * 180.0 / M_PI, error.path_curvature);
}

void MPPIHCRos::publishCommand(const BodyVelocity& cmd)
{
    geometry_msgs::Twist twist;
    twist.linear.x = cmd.vx;
    twist.linear.y = cmd.vy;
    twist.angular.z = cmd.omega;
    cmd_vel_pub_.publish(twist);
}

void MPPIHCRos::publishVisualization()
{
    // Publish optimal trajectory
    const auto& opt_traj = controller_->getOptimalTrajectory();
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

    // Publish status overlay
    const auto& stats = controller_->getEstimatorStats();
    jsk_rviz_plugins::OverlayText overlay;
    overlay.action = jsk_rviz_plugins::OverlayText::ADD;
    overlay.width = 300;
    overlay.height = 150;
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
    ss << "=== MPPI-HC Status ===" << std::endl;
    ss << "K_slip:     " << stats.current_k_slip << std::endl;
    ss << "K_slip_raw: " << stats.raw_k_slip << std::endl;
    ss << "Error:      " << stats.estimation_error << std::endl;
    ss << "Converged:  " << (stats.is_converged ? "Yes" : "No") << std::endl;
    ss << "Samples:    " << stats.num_samples << std::endl;
    overlay.text = ss.str();

    overlay_text_pub_.publish(overlay);
}

void MPPIHCRos::publishEvalMessage()
{
    mppi_eval_msgs::MPPIEval eval;
    eval.header.stamp = ros::Time::now();
    eval.header.frame_id = "odom";

    // Current state (using existing message fields)
    eval.global_x = current_state_.x;
    eval.global_y = current_state_.y;
    eval.global_yaw = current_state_.yaw;
    eval.cmd_vx = current_state_.vx;
    eval.cmd_vy = current_state_.vy;
    eval.cmd_yawrate = current_state_.omega;

    // State cost (use as slip factor for now)
    const auto& stats = controller_->getEstimatorStats();
    eval.state_cost = stats.current_k_slip;

    // Goal reached check
    double pos_error = std::hypot(
        goal_state_.x - current_state_.x,
        goal_state_.y - current_state_.y);
    eval.goal_reached = (pos_error < 0.1);  // 10cm threshold

    eval_msg_pub_.publish(eval);
}

MPPIHCRos::TrackingError MPPIHCRos::computeTrackingError() const
{
    TrackingError error;
    error.lateral_error = 0.0;
    error.heading_error = 0.0;
    error.path_curvature = 0.0;
    error.closest_idx = 0;

    if (ref_path_.poses.size() < 2) {
        return error;
    }

    // Find closest point on path
    double min_dist = std::numeric_limits<double>::max();
    int closest_idx = 0;
    
    for (size_t i = 0; i < ref_path_.poses.size(); ++i) {
        double dx = ref_path_.poses[i].pose.position.x - current_state_.x;
        double dy = ref_path_.poses[i].pose.position.y - current_state_.y;
        double dist = std::hypot(dx, dy);
        if (dist < min_dist) {
            min_dist = dist;
            closest_idx = static_cast<int>(i);
        }
    }
    error.closest_idx = closest_idx;

    // Get path heading at closest point
    double path_yaw = tf2::getYaw(ref_path_.poses[closest_idx].pose.orientation);

    // Cross-track error (signed)
    // Positive = robot is to the LEFT of the path
    double dx = current_state_.x - ref_path_.poses[closest_idx].pose.position.x;
    double dy = current_state_.y - ref_path_.poses[closest_idx].pose.position.y;
    error.lateral_error = -dx * std::sin(path_yaw) + dy * std::cos(path_yaw);

    // Heading error
    error.heading_error = current_state_.yaw - path_yaw;
    // Normalize to [-pi, pi]
    while (error.heading_error > M_PI) error.heading_error -= 2.0 * M_PI;
    while (error.heading_error < -M_PI) error.heading_error += 2.0 * M_PI;

    // Compute local curvature using 3 points
    int n = static_cast<int>(ref_path_.poses.size());
    int i0 = std::max(0, closest_idx - 3);
    int i2 = std::min(n - 1, closest_idx + 3);
    
    if (i2 > i0 + 1) {
        double x0 = ref_path_.poses[i0].pose.position.x;
        double y0 = ref_path_.poses[i0].pose.position.y;
        double x1 = ref_path_.poses[closest_idx].pose.position.x;
        double y1 = ref_path_.poses[closest_idx].pose.position.y;
        double x2 = ref_path_.poses[i2].pose.position.x;
        double y2 = ref_path_.poses[i2].pose.position.y;

        // Menger curvature
        double area2 = std::abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
        double d01 = std::hypot(x1 - x0, y1 - y0);
        double d12 = std::hypot(x2 - x1, y2 - y1);
        double d20 = std::hypot(x0 - x2, y0 - y2);
        double denom = d01 * d12 * d20;

        if (denom > 1e-6) {
            error.path_curvature = area2 / denom;
            // Sign: positive = left turn
            double cross = (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1);
            error.path_curvature = std::copysign(error.path_curvature, cross);
        }
    }

    return error;
}

} // namespace mppi_hc
