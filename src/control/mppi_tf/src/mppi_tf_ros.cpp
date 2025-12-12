/**
 * @file mppi_tf_ros.cpp
 * @brief ROS wrapper implementation for MPPI-TireForce controller
 * 
 * @author ZZT
 * @date 2024
 */

#include "mppi_tf/mppi_tf_ros.hpp"
#include <tf2/utils.h>
#include <iomanip>
#include <sstream>

namespace mppi_tf
{

MPPITFRos::MPPITFRos(ros::NodeHandle& nh, ros::NodeHandle& private_nh)
    : nh_(nh), private_nh_(private_nh), tf_listener_(tf_buffer_)
{
    // Load parameters
    loadParameters();

    // Create controller
    controller_ = std::make_unique<MPPITFCore>(config_);
    
    // Set feedback gains
    controller_->setFeedbackGains(
        feedback_gains_[0], feedback_gains_[1], feedback_gains_[2]);
    
    ROS_INFO("[MPPI-TF] Feedback gains: k_lat=%.2f, k_head=%.2f, k_int=%.2f",
             feedback_gains_[0], feedback_gains_[1], feedback_gains_[2]);

    // Set sampling mode
    std::string sampling_mode;
    private_nh_.param<std::string>("sampling_mode", sampling_mode, "velocity");
    if (sampling_mode == "force") {
        controller_->setSamplingMode(SamplingMode::FORCE_SPACE);
        ROS_INFO("[MPPI-TF] Using FORCE-SPACE sampling");
    } else {
        controller_->setSamplingMode(SamplingMode::VELOCITY_SPACE);
        ROS_INFO("[MPPI-TF] Using VELOCITY-SPACE sampling");
    }

    // Subscribers
    std::string odom_topic, path_topic, collision_topic, distance_topic, yaw_topic;
    private_nh_.param<std::string>("odom_topic", odom_topic, "/groundtruth_odom");
    private_nh_.param<std::string>("ref_path_topic", path_topic, "/move_base/NavfnROS/plan");
    private_nh_.param<std::string>("collision_costmap_topic", collision_topic, "/move_base/local_costmap/costmap");
    private_nh_.param<std::string>("distance_error_map_topic", distance_topic, "/distance_error_map");
    private_nh_.param<std::string>("ref_yaw_map_topic", yaw_topic, "/ref_yaw_map");

    odom_sub_ = nh_.subscribe(odom_topic, 1, &MPPITFRos::odomCallback, this);
    ref_path_sub_ = nh_.subscribe(path_topic, 1, &MPPITFRos::refPathCallback, this);
    collision_costmap_sub_ = nh_.subscribe(collision_topic, 1, &MPPITFRos::collisionCostmapCallback, this);
    distance_error_map_sub_ = nh_.subscribe(distance_topic, 1, &MPPITFRos::distanceErrorMapCallback, this);
    ref_yaw_map_sub_ = nh_.subscribe(yaw_topic, 1, &MPPITFRos::refYawMapCallback, this);

    // Publishers
    std::string cmd_topic;
    private_nh_.param<std::string>("cmd_vel_topic", cmd_topic, "/cmd_vel");
    
    cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>(cmd_topic, 1);
    optimal_traj_pub_ = nh_.advertise<nav_msgs::Path>("mppi_tf/optimal_trajectory", 1);
    sampled_traj_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("mppi_tf/sampled_trajectories", 1);
    calc_time_pub_ = nh_.advertise<std_msgs::Float32>("mppi_tf/calc_time", 1);
    overlay_text_pub_ = nh_.advertise<jsk_rviz_plugins::OverlayText>("mppi_tf/status", 1);
    eval_msg_pub_ = nh_.advertise<mppi_eval_msgs::MPPIEval>("mppi_tf/eval", 1);
    friction_circle_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("mppi_tf/friction_circles", 1);

    // Control timer
    control_timer_ = nh_.createTimer(
        ros::Duration(1.0 / config_.control_rate), 
        &MPPITFRos::controlTimerCallback, this);

    ROS_INFO("[MPPI-TF] Controller initialized successfully");
    ROS_INFO("[MPPI-TF] Tire Model: Pacejka Magic Formula");
    ROS_INFO("[MPPI-TF] Samples: %d, Horizon: %d, dt: %.3f",
             config_.mppi.num_samples, config_.mppi.prediction_horizon, config_.mppi.step_dt);
}

void MPPITFRos::loadParameters()
{
    // Vehicle geometry
    private_nh_.param("vehicle/l_f", config_.vehicle.geometry.l_f, 0.5);
    private_nh_.param("vehicle/l_r", config_.vehicle.geometry.l_r, 0.5);
    private_nh_.param("vehicle/d_l", config_.vehicle.geometry.d_l, 0.5);
    private_nh_.param("vehicle/d_r", config_.vehicle.geometry.d_r, 0.5);
    private_nh_.param("vehicle/tire_radius", config_.vehicle.geometry.tire_radius, 0.2);

    // Vehicle mass
    private_nh_.param("vehicle/mass", config_.vehicle.mass.mass, 100.0);
    private_nh_.param("vehicle/Iz", config_.vehicle.mass.Iz, 50.0);
    private_nh_.param("vehicle/h_cog", config_.vehicle.mass.h_cog, 0.3);

    // Tire parameters
    private_nh_.param("tire/C_alpha", config_.vehicle.tire.C_alpha, 50000.0);
    private_nh_.param("tire/C_kappa", config_.vehicle.tire.C_kappa, 100000.0);
    private_nh_.param("tire/mu_peak", config_.vehicle.tire.mu_peak, 0.9);
    private_nh_.param("tire/mu_slide", config_.vehicle.tire.mu_slide, 0.7);
    private_nh_.param("tire/B", config_.vehicle.tire.B, 10.0);
    private_nh_.param("tire/C", config_.vehicle.tire.C, 1.9);
    private_nh_.param("tire/E", config_.vehicle.tire.E, 0.97);

    // Velocity limits
    private_nh_.param("vehicle/vx_max", config_.vehicle.limits.vx_max, 3.0);
    private_nh_.param("vehicle/vy_max", config_.vehicle.limits.vy_max, 3.0);
    private_nh_.param("vehicle/omega_max", config_.vehicle.limits.omega_max, 3.0);
    private_nh_.param("vehicle/ax_max", config_.vehicle.limits.ax_max, 5.0);
    private_nh_.param("vehicle/ay_max", config_.vehicle.limits.ay_max, 5.0);

    // MPPI parameters
    private_nh_.param("mppi/num_samples", config_.mppi.num_samples, 3000);
    private_nh_.param("mppi/horizon", config_.mppi.prediction_horizon, 40);
    private_nh_.param("mppi/dt", config_.mppi.step_dt, 0.033);
    private_nh_.param("mppi/lambda", config_.mppi.lambda, 100.0);
    private_nh_.param("mppi/alpha", config_.mppi.alpha, 0.975);
    private_nh_.param("mppi/exploration_ratio", config_.mppi.exploration_ratio, 0.1);
    private_nh_.param("mppi/ref_velocity", config_.mppi.ref_velocity, 2.0);

    // Noise parameters
    double sigma_fx, sigma_fy, sigma_mz;
    private_nh_.param("mppi/sigma_fx", sigma_fx, 200.0);
    private_nh_.param("mppi/sigma_fy", sigma_fy, 200.0);
    private_nh_.param("mppi/sigma_mz", sigma_mz, 50.0);
    config_.mppi.sigma_force = Eigen::Vector3d(sigma_fx, sigma_fy, sigma_mz);

    double sigma_vx, sigma_vy, sigma_omega;
    private_nh_.param("mppi/sigma_vx", sigma_vx, 0.55);
    private_nh_.param("mppi/sigma_vy", sigma_vy, 0.55);
    private_nh_.param("mppi/sigma_omega", sigma_omega, 0.96);
    config_.mppi.sigma_vel = Eigen::Vector3d(sigma_vx, sigma_vy, sigma_omega);

    // Cost weights
    private_nh_.param("cost/distance_error", config_.weights.distance_error, 60.0);
    private_nh_.param("cost/angular_error", config_.weights.angular_error, 30.0);
    private_nh_.param("cost/velocity_error", config_.weights.velocity_error, 10.0);
    private_nh_.param("cost/terminal_state", config_.weights.terminal_state, 10.0);
    private_nh_.param("cost/collision", config_.weights.collision_penalty, 100.0);
    private_nh_.param("cost/force_utilization", config_.weights.force_utilization, 20.0);
    private_nh_.param("cost/friction_margin", config_.weights.friction_margin, 30.0);
    private_nh_.param("cost/slip_angle", config_.weights.slip_angle, 15.0);
    private_nh_.param("cost/curvature_speed", config_.weights.curvature_speed, 40.0);
    private_nh_.param("cost/force_change", config_.weights.force_change, 5.0);

    // Estimator parameters
    private_nh_.param("estimator/lr_cornering", config_.estimator.lr_cornering, 0.01);
    private_nh_.param("estimator/lr_friction", config_.estimator.lr_friction, 0.005);
    private_nh_.param("estimator/C_alpha_min", config_.estimator.C_alpha_min, 10000.0);
    private_nh_.param("estimator/C_alpha_max", config_.estimator.C_alpha_max, 100000.0);
    private_nh_.param("estimator/mu_min", config_.estimator.mu_min, 0.1);
    private_nh_.param("estimator/mu_max", config_.estimator.mu_max, 1.2);

    // Feedback gains
    private_nh_.param("feedback/k_lateral", feedback_gains_[0], 1.5);
    private_nh_.param("feedback/k_heading", feedback_gains_[1], 0.5);
    private_nh_.param("feedback/k_integral", feedback_gains_[2], 0.2);

    // SG filter
    private_nh_.param("sg_filter/enable", config_.sg_filter.enable, true);
    private_nh_.param("sg_filter/half_window", config_.sg_filter.half_window, 4);
    private_nh_.param("sg_filter/poly_order", config_.sg_filter.poly_order, 3);

    // Goal tolerance
    private_nh_.param("xy_goal_tolerance", config_.xy_goal_tolerance, 0.5);
    private_nh_.param("yaw_goal_tolerance", config_.yaw_goal_tolerance, 3.14);

    // Control rate
    private_nh_.param("control_rate", config_.control_rate, 50.0);

    // Goal proximity
    private_nh_.param("goal_proximity/threshold", goal_proximity_threshold_, 0.8);
    private_nh_.param("goal_proximity/feedback_fade", goal_feedback_fade_, true);

    ROS_INFO("[MPPI-TF] Parameters loaded");
}

// ============================================================================
// Callbacks
// ============================================================================

void MPPITFRos::odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
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
    static BodyVelocity last_cmd;
    BodyVelocity actual_vel{current_state_.vx, current_state_.vy, current_state_.omega};
    controller_->updateEstimator(last_cmd, actual_vel, 0.02);
}

void MPPITFRos::refPathCallback(const nav_msgs::Path::ConstPtr& msg)
{
    if (msg->poses.empty()) {
        ROS_WARN_THROTTLE(1.0, "[MPPI-TF] Received empty path");
        return;
    }

    ref_path_ = *msg;

    const auto& goal_pose = msg->poses.back().pose;
    goal_state_.x = goal_pose.position.x;
    goal_state_.y = goal_pose.position.y;
    goal_state_.yaw = tf2::getYaw(goal_pose.orientation);
    goal_state_.vx = 0.0;
    goal_state_.vy = 0.0;
    goal_state_.omega = 0.0;

    path_received_ = true;
}

void MPPITFRos::collisionCostmapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg)
{
    grid_map::GridMapRosConverter::fromOccupancyGrid(*msg, "collision_cost", collision_map_);
    costmaps_received_ = true;
}

void MPPITFRos::distanceErrorMapCallback(const grid_map_msgs::GridMap::ConstPtr& msg)
{
    grid_map::GridMapRosConverter::fromMessage(*msg, distance_error_map_);
}

void MPPITFRos::refYawMapCallback(const grid_map_msgs::GridMap::ConstPtr& msg)
{
    grid_map::GridMapRosConverter::fromMessage(*msg, ref_yaw_map_);
}

void MPPITFRos::controlTimerCallback(const ros::TimerEvent& event)
{
    if (!odom_received_ || !path_received_ || !costmaps_received_) {
        ROS_WARN_THROTTLE(1.0, "[MPPI-TF] Waiting for data... odom:%d path:%d maps:%d",
                         odom_received_, path_received_, costmaps_received_);
        return;
    }

    auto start_time = ros::Time::now();

    // Compute dt
    double dt = 0.02;
    if (!last_control_time_.isZero()) {
        dt = (start_time - last_control_time_).toSec();
        dt = std::clamp(dt, 0.001, 0.1);
    }
    last_control_time_ = start_time;

    // Compute tracking error
    TrackingError error = computeTrackingError();

    // Distance to goal
    double dist_to_goal = std::hypot(goal_state_.x - current_state_.x,
                                      goal_state_.y - current_state_.y);

    // Safety checks and feedback scaling
    double current_speed = current_state_.speed();
    
    // Low speed: reduce feedback
    const double low_speed_threshold = 0.3;
    if (current_speed < low_speed_threshold) {
        double speed_factor = current_speed / low_speed_threshold;
        error.lateral_error *= speed_factor;
        error.heading_error *= speed_factor;
    }

    // Large error: cap feedback
    const double large_error_threshold = 0.5;
    if (std::abs(error.lateral_error) > large_error_threshold) {
        double error_factor = large_error_threshold / std::abs(error.lateral_error);
        error.lateral_error *= error_factor;
    }

    // Near goal: fade feedback
    if (goal_feedback_fade_ && dist_to_goal < goal_proximity_threshold_) {
        double fade = dist_to_goal / goal_proximity_threshold_;
        error.lateral_error *= fade;
        error.heading_error *= fade;
    }

    // Solve MPPI
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
    publishEvalMessage();
    publishFrictionCircle();

    // Publish timing
    std_msgs::Float32 time_msg;
    time_msg.data = controller_->getCalcTimeMs();
    calc_time_pub_.publish(time_msg);

    ROS_INFO_THROTTLE(2.0, "[MPPI-TF] lat_err: %.3fm, C_alpha: %.0f, mu: %.2f, time: %.1fms",
                     error.lateral_error,
                     controller_->getCorneringStiffness(),
                     controller_->getFrictionCoeff(),
                     controller_->getCalcTimeMs());
}

// ============================================================================
// Publishing
// ============================================================================

void MPPITFRos::publishCommand(const BodyVelocity& cmd)
{
    geometry_msgs::Twist twist;
    twist.linear.x = cmd.vx;
    twist.linear.y = cmd.vy;
    twist.angular.z = cmd.omega;
    cmd_vel_pub_.publish(twist);
}

void MPPITFRos::publishVisualization()
{
    // Optimal trajectory
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

    // Overlay text
    const auto& stats = controller_->getEstimatorStats();
    jsk_rviz_plugins::OverlayText overlay;
    overlay.action = jsk_rviz_plugins::OverlayText::ADD;
    overlay.width = 350;
    overlay.height = 200;
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
    ss << "===== MPPI-TireForce =====" << std::endl;
    ss << "Mode: " << (controller_->getSamplingMode() == SamplingMode::FORCE_SPACE ? "Force" : "Velocity") << std::endl;
    ss << "------------------------" << std::endl;
    ss << "C_alpha: " << std::setprecision(0) << stats.C_alpha_est << " N/rad" << std::endl;
    ss << "mu_est:  " << std::setprecision(3) << stats.mu_est << std::endl;
    ss << "Error:   " << stats.estimation_error << std::endl;
    ss << "Converged: " << (stats.is_converged ? "Yes" : "No") << std::endl;
    ss << "------------------------" << std::endl;
    ss << "Calc time: " << std::setprecision(1) << controller_->getCalcTimeMs() << " ms" << std::endl;
    ss << "Cost:      " << std::setprecision(2) << controller_->getStateCost() << std::endl;
    overlay.text = ss.str();

    overlay_text_pub_.publish(overlay);
}

void MPPITFRos::publishEvalMessage()
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

    const auto& stats = controller_->getEstimatorStats();
    eval.state_cost = stats.C_alpha_est;

    double pos_error = std::hypot(goal_state_.x - current_state_.x,
                                  goal_state_.y - current_state_.y);
    eval.goal_reached = (pos_error < 0.1);

    eval_msg_pub_.publish(eval);
}

void MPPITFRos::publishFrictionCircle()
{
    visualization_msgs::MarkerArray markers;
    
    // Get wheel commands for visualization
    VehicleCommand8D wheel_cmd = controller_->getWheelCommands();
    
    // Colors for each wheel
    std::array<std::array<float, 3>, 4> colors = {{
        {1.0f, 0.0f, 0.0f},   // FL - Red
        {0.0f, 1.0f, 0.0f},   // FR - Green
        {0.0f, 0.0f, 1.0f},   // RL - Blue
        {1.0f, 1.0f, 0.0f}    // RR - Yellow
    }};
    
    double mu = controller_->getFrictionCoeff();
    
    for (int i = 0; i < NUM_WHEELS; ++i) {
        Position2D wheel_pos = config_.vehicle.geometry.getWheelPosition(static_cast<WheelIndex>(i));
        double Fz = config_.vehicle.mass.getStaticWheelLoad(static_cast<WheelIndex>(i));
        double F_max = mu * Fz;
        
        // Friction circle
        visualization_msgs::Marker circle;
        circle.header.frame_id = "base_link";
        circle.header.stamp = ros::Time::now();
        circle.ns = "friction_circles";
        circle.id = i;
        circle.type = visualization_msgs::Marker::LINE_STRIP;
        circle.action = visualization_msgs::Marker::ADD;
        circle.scale.x = 0.01;
        circle.color.r = colors[i][0];
        circle.color.g = colors[i][1];
        circle.color.b = colors[i][2];
        circle.color.a = 0.5;
        
        // Draw circle at wheel position
        double radius = F_max / 5000.0;  // Scale for visualization
        for (int j = 0; j <= 36; ++j) {
            double angle = 2.0 * M_PI * j / 36.0;
            geometry_msgs::Point p;
            p.x = wheel_pos.x + radius * std::cos(angle);
            p.y = wheel_pos.y + radius * std::sin(angle);
            p.z = 0.05;
            circle.points.push_back(p);
        }
        markers.markers.push_back(circle);
    }
    
    friction_circle_pub_.publish(markers);
}

// ============================================================================
// Utilities
// ============================================================================

TrackingError MPPITFRos::computeTrackingError() const
{
    TrackingError error;
    
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

    // Compute lateral error (cross-track error)
    // Use vector from closest point to robot, project onto path normal
    if (closest_idx < static_cast<int>(ref_path_.poses.size()) - 1) {
        double x0 = ref_path_.poses[closest_idx].pose.position.x;
        double y0 = ref_path_.poses[closest_idx].pose.position.y;
        double x1 = ref_path_.poses[closest_idx + 1].pose.position.x;
        double y1 = ref_path_.poses[closest_idx + 1].pose.position.y;

        // Path tangent
        double tx = x1 - x0;
        double ty = y1 - y0;
        double path_len = std::hypot(tx, ty);

        if (path_len > EPSILON) {
            tx /= path_len;
            ty /= path_len;

            // Vector from path point to robot
            double dx = current_state_.x - x0;
            double dy = current_state_.y - y0;

            // Cross product gives signed distance (positive = robot left of path)
            error.lateral_error = tx * dy - ty * dx;
        }

        // Heading error
        double path_yaw = std::atan2(ty, tx);
        error.heading_error = std::remainder(current_state_.yaw - path_yaw, 2.0 * M_PI);

        // Estimate curvature from path
        if (closest_idx < static_cast<int>(ref_path_.poses.size()) - 2) {
            double x2 = ref_path_.poses[closest_idx + 2].pose.position.x;
            double y2 = ref_path_.poses[closest_idx + 2].pose.position.y;

            double dx1 = x1 - x0;
            double dy1 = y1 - y0;
            double dx2 = x2 - x1;
            double dy2 = y2 - y1;

            double yaw1 = std::atan2(dy1, dx1);
            double yaw2 = std::atan2(dy2, dx2);
            double dyaw = std::remainder(yaw2 - yaw1, 2.0 * M_PI);
            double ds = std::hypot(dx1, dy1);

            if (ds > EPSILON) {
                error.path_curvature = std::abs(dyaw) / ds;
            }
        }
    }

    return error;
}

} // namespace mppi_tf
