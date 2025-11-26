#pragma once

/**
 * @file mppi_hc_ros.hpp
 * @brief ROS wrapper for MPPI-HC controller
 */

#include "mppi_hc/mppi_hc_core.hpp"

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/Twist.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/Float32.h>
#include <jsk_rviz_plugins/OverlayText.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <tf2_ros/transform_listener.h>
#include <mppi_eval_msgs/MPPIEval.h>

namespace mppi_hc
{

/**
 * @brief ROS interface for MPPI-HC controller
 */
class MPPIHCRos
{
public:
    MPPIHCRos(ros::NodeHandle& nh, ros::NodeHandle& private_nh);
    ~MPPIHCRos() = default;

private:
    // ROS handles
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

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

    // Timer
    ros::Timer control_timer_;

    // TF
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // Core controller
    std::unique_ptr<MPPIHCCore> controller_;
    ControllerConfig config_;

    // State
    State current_state_;
    State goal_state_;
    bool odom_received_ = false;
    bool path_received_ = false;
    bool costmaps_received_ = false;

    // Grid maps
    grid_map::GridMap collision_map_;
    grid_map::GridMap distance_error_map_;
    grid_map::GridMap ref_yaw_map_;

    // Callbacks
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void refPathCallback(const nav_msgs::Path::ConstPtr& msg);
    void collisionCostmapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg);
    void distanceErrorMapCallback(const grid_map_msgs::GridMap::ConstPtr& msg);
    void refYawMapCallback(const grid_map_msgs::GridMap::ConstPtr& msg);
    void controlTimerCallback(const ros::TimerEvent& event);

    // Helpers
    void loadParameters();
    void publishCommand(const BodyVelocity& cmd);
    void publishVisualization();
    void publishEvalMessage();
};

} // namespace mppi_hc
