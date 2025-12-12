#pragma once

/**
 * @file mppi_tf_ros.hpp
 * @brief ROS wrapper for MPPI-TireForce controller
 * 
 * This class handles:
 * - ROS parameter loading
 * - Subscribers for odometry, path, costmaps
 * - Publishers for commands, visualization, diagnostics
 * - Timing and control loop
 * - Tracking error computation
 * 
 * @author ZZT
 * @date 2024
 */

#include "mppi_tf/mppi_tf_core.hpp"

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/OccupancyGrid.h>
#include <std_msgs/Float32.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>

#include <jsk_rviz_plugins/OverlayText.h>
#include <mppi_eval_msgs/MPPIEval.h>

#include <memory>

namespace mppi_tf
{

/**
 * @brief Tracking error structure
 */
struct TrackingError
{
    double lateral_error = 0.0;   // Cross-track error [m]
    double heading_error = 0.0;   // Heading error [rad]
    double path_curvature = 0.0;  // Local curvature [1/m]
    int closest_idx = 0;          // Closest path point index
};

/**
 * @brief ROS wrapper for MPPI-TireForce controller
 */
class MPPITFRos
{
public:
    MPPITFRos(ros::NodeHandle& nh, ros::NodeHandle& private_nh);
    ~MPPITFRos() = default;

private:
    // ROS handles
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

    // Core controller
    std::unique_ptr<MPPITFCore> controller_;
    ControllerConfig config_;

    // TF
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // Subscribers
    ros::Subscriber odom_sub_;
    ros::Subscriber ref_path_sub_;
    ros::Subscriber collision_costmap_sub_;
    ros::Subscriber distance_error_map_sub_;
    ros::Subscriber ref_yaw_map_sub_;

    // Publishers
    ros::Publisher cmd_vel_pub_;
    ros::Publisher optimal_traj_pub_;
    ros::Publisher sampled_traj_pub_;
    ros::Publisher calc_time_pub_;
    ros::Publisher overlay_text_pub_;
    ros::Publisher eval_msg_pub_;
    ros::Publisher friction_circle_pub_;

    // Timer
    ros::Timer control_timer_;

    // State
    FullState current_state_;
    FullState goal_state_;
    nav_msgs::Path ref_path_;
    grid_map::GridMap collision_map_;
    grid_map::GridMap distance_error_map_;
    grid_map::GridMap ref_yaw_map_;

    // Flags
    bool odom_received_ = false;
    bool path_received_ = false;
    bool costmaps_received_ = false;

    // Feedback gains (loaded from params)
    std::array<double, 3> feedback_gains_;

    // Goal proximity settings
    double goal_proximity_threshold_ = 0.8;
    bool goal_feedback_fade_ = true;

    // Last control time
    ros::Time last_control_time_;

    // ========================================================================
    // Methods
    // ========================================================================

    void loadParameters();

    // Callbacks
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void refPathCallback(const nav_msgs::Path::ConstPtr& msg);
    void collisionCostmapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg);
    void distanceErrorMapCallback(const grid_map_msgs::GridMap::ConstPtr& msg);
    void refYawMapCallback(const grid_map_msgs::GridMap::ConstPtr& msg);
    void controlTimerCallback(const ros::TimerEvent& event);

    // Publishing
    void publishCommand(const BodyVelocity& cmd);
    void publishVisualization();
    void publishEvalMessage();
    void publishFrictionCircle();

    // Utilities
    TrackingError computeTrackingError() const;
};

} // namespace mppi_tf
