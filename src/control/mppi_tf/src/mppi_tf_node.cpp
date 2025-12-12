/**
 * @file mppi_tf_node.cpp
 * @brief Main ROS node for MPPI-TireForce controller
 * 
 * @author ZZT
 * @date 2024
 */

#include "mppi_tf/mppi_tf_ros.hpp"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "mppi_tf_node");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    try {
        mppi_tf::MPPITFRos controller(nh, private_nh);
        ros::spin();
    } catch (const std::exception& e) {
        ROS_FATAL("[MPPI-TF] Exception: %s", e.what());
        return 1;
    }

    return 0;
}
