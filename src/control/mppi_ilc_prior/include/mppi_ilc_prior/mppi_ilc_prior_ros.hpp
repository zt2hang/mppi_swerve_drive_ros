#pragma once

#include "mppi_ilc_prior/mppi_ilc_prior_core.hpp"

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

#include <fstream>
#include <vector>

namespace mppi_ilc_prior
{

class MPPIILCPriorRos
{
public:
    MPPIILCPriorRos(ros::NodeHandle& nh, ros::NodeHandle& private_nh);
    ~MPPIILCPriorRos() = default;

private:
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

    ros::Subscriber odom_sub_;
    ros::Subscriber ref_path_sub_;
    ros::Subscriber collision_costmap_sub_;
    ros::Subscriber distance_error_map_sub_;
    ros::Subscriber ref_yaw_map_sub_;

    ros::Publisher cmd_vel_pub_;
    ros::Publisher optimal_traj_pub_;
    ros::Publisher sampled_traj_pub_;
    ros::Publisher calc_time_pub_;
    ros::Publisher overlay_text_pub_;
    ros::Publisher eval_msg_pub_;

    ros::Timer control_timer_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    std::unique_ptr<MPPIILCPriorCore> controller_;
    mppi_hc::ControllerConfig config_;
    ILCPriorLearningConfig ilc_cfg_;
    bool ilc_enabled_ = true;
    bool reset_ilc_on_new_path_ = true;
    double feedback_gains_[3] = {2.5, 1.0, 0.5};

    mppi_hc::State current_state_;
    mppi_hc::State goal_state_;
    bool odom_received_ = false;
    bool path_received_ = false;
    bool costmaps_received_ = false;

    grid_map::GridMap collision_map_;
    grid_map::GridMap distance_error_map_;
    grid_map::GridMap ref_yaw_map_;
    nav_msgs::Path ref_path_;
    ros::Time last_control_time_;

    // Path arclength (monotonic along ref_path_ order)
    std::vector<double> ref_path_cum_s_;
    bool ref_path_cum_s_ready_ = false;

    // Closest-index continuity (avoid branch jumping on self-intersections)
    int last_closest_idx_ = -1;
    bool have_last_closest_idx_ = false;
    bool path_is_closed_ = false;

    int idx_window_back_ = 25;
    int idx_window_fwd_ = 60;
    bool idx_allow_wraparound_ = true;
    double idx_closed_path_threshold_ = 0.6;   // [m] start/end distance to treat as closed
    double idx_heading_weight_ = 0.4;          // [m/rad] distance penalty per heading mismatch
    double idx_heading_gate_ = 1.2;            // [rad] reject window result if too misaligned
    double idx_global_fallback_factor_ = 1.6;  // if window_dist > factor*global_dist => fallback

    double goal_proximity_threshold_ = 0.8;
    bool goal_feedback_fade_ = true;

    // Prior indexing mode
    bool prior_use_arclength_ = true;
    double prior_speed_min_ = 0.2;  // [m/s]

    // Metrics / lap statistics (closed-loop only)
    bool metrics_enabled_ = true;
    double metrics_corner_curvature_threshold_ = 0.10;
    double metrics_wrap_s_margin_ = 0.5;  // [m] hysteresis for lap wrap detection

    double metrics_last_s_ = 0.0;
    bool metrics_have_last_s_ = false;
    int metrics_lap_count_ = 0;
    std::size_t metrics_samples_total_ = 0;
    std::size_t metrics_samples_corner_ = 0;
    std::size_t metrics_samples_straight_ = 0;
    double metrics_sum_lat2_total_ = 0.0;
    double metrics_sum_lat2_corner_ = 0.0;
    double metrics_sum_lat2_straight_ = 0.0;
    double metrics_sum_head2_total_ = 0.0;

    std::size_t metrics_ilc_updates_ = 0;
    std::size_t metrics_ilc_saturation_v_ = 0;
    std::size_t metrics_ilc_saturation_omega_ = 0;
    double metrics_sum_delta_v2_ = 0.0;
    double metrics_sum_delta_omega2_ = 0.0;

    // Metrics logging (CSV; Excel-friendly)
    bool metrics_log_to_file_ = true;
    std::string metrics_log_dir_;
    std::string metrics_log_tag_;
    bool metrics_log_inited_ = false;
    std::ofstream metrics_lap_csv_;

    struct TrackingError {
        double lateral_error = 0.0;
        double heading_error = 0.0;
        double path_curvature = 0.0;
        int closest_idx = 0;
    };

    void initMetricsLogIfNeeded();
    void writeLapMetricsCsv(
        int lap,
        double path_length,
        double rmse_lat,
        double rmse_lat_straight,
        double rmse_lat_corner,
        double rmse_head_deg,
        std::size_t samples_total,
        const ILCMemory::BiasStats& bias_stats,
        std::size_t ilc_updates,
        double delta_v_rms,
        double delta_w_rms,
        std::size_t sat_v,
        std::size_t sat_w);

    void resetLapMetrics();
    void updateLapMetrics(const TrackingError& e);
    void maybeFinishLap(double current_s, double path_length);
    void printLapMetrics(double path_length);

    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void refPathCallback(const nav_msgs::Path::ConstPtr& msg);
    void collisionCostmapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg);
    void distanceErrorMapCallback(const grid_map_msgs::GridMap::ConstPtr& msg);
    void refYawMapCallback(const grid_map_msgs::GridMap::ConstPtr& msg);
    void controlTimerCallback(const ros::TimerEvent& event);

    void loadParameters();
    void publishCommand(const mppi_hc::BodyVelocity& cmd);
    void publishVisualization();
    void publishEvalMessage();
    TrackingError computeTrackingError() const;
};

}  // namespace mppi_ilc_prior
