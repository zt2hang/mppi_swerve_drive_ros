#include "mppi_ilc/mppi_ilc_ros.hpp"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "mppi_ilc_node");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    mppi_ilc::MPPIILCRos node(nh, private_nh);
    ros::spin();
    return 0;
}
