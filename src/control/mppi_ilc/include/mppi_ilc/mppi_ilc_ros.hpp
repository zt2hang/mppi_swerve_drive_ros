#pragma once

#include "mppi_ilc/mppi_ilc_core.hpp"

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

namespace mppi_ilc
{

class MPPIILCRos
{
public:
    MPPIILCRos(ros::NodeHandle& nh, ros::NodeHandle& private_nh);
    ~MPPIILCRos() = default;

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
    std::unique_ptr<MPPIILCCore> controller_;
    mppi_hc::ControllerConfig config_;
    ILCLearningConfig ilc_cfg_;
    bool ilc_enabled_ = true;
    bool reset_ilc_on_new_path_ = true;
    double feedback_gains_[3] = {2.5, 1.0, 0.5};

    // State
    mppi_hc::State current_state_;
    mppi_hc::State goal_state_;
    bool odom_received_ = false;
    bool path_received_ = false;
    bool costmaps_received_ = false;

    // Grid maps
    grid_map::GridMap collision_map_;
    grid_map::GridMap distance_error_map_;
    grid_map::GridMap ref_yaw_map_;
    nav_msgs::Path ref_path_;
    ros::Time last_control_time_;

    // Closest-index continuity (avoid branch jumping on self-intersections)
    int last_closest_idx_ = -1;
    bool have_last_closest_idx_ = false;
    bool path_is_closed_ = false;

    int idx_window_back_ = 25;
    int idx_window_fwd_ = 60;
    bool idx_allow_wraparound_ = true;
    double idx_closed_path_threshold_ = 0.6;   // [m]
    double idx_heading_weight_ = 0.4;          // [m/rad]
    double idx_heading_gate_ = 1.2;            // [rad]
    double idx_global_fallback_factor_ = 1.6;

    double goal_proximity_threshold_ = 0.8;
    bool goal_feedback_fade_ = true;

    // Callbacks
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void refPathCallback(const nav_msgs::Path::ConstPtr& msg);
    void collisionCostmapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg);
    void distanceErrorMapCallback(const grid_map_msgs::GridMap::ConstPtr& msg);
    void refYawMapCallback(const grid_map_msgs::GridMap::ConstPtr& msg);
    void controlTimerCallback(const ros::TimerEvent& event);

    // Helpers
    void loadParameters();
    void publishCommand(const mppi_hc::BodyVelocity& cmd);
    void publishVisualization();
    void publishEvalMessage();

    struct TrackingError {
        double lateral_error = 0.0;
        double heading_error = 0.0;
        double path_curvature = 0.0;
        int closest_idx = 0;
    };
    TrackingError computeTrackingError() const;
};

}  // namespace mppi_ilc
