/**
 * @file mppi_hc_node.cpp
 * @brief ROS node entry point for MPPI-HC controller
 */

#include <ros/ros.h>
#include "mppi_hc/mppi_hc_ros.hpp"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "mppi_hc_node");
    
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    try {
        mppi_hc::MPPIHCRos controller(nh, private_nh);
        ros::spin();
    }
    catch (const std::exception& e) {
        ROS_FATAL("[MPPI-HC] Exception caught: %s", e.what());
        return 1;
    }

    return 0;
}
