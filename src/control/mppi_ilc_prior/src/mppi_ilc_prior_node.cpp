#include "mppi_ilc_prior/mppi_ilc_prior_ros.hpp"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "mppi_ilc_prior_node");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    mppi_ilc_prior::MPPIILCPriorRos node(nh, private_nh);
    ros::spin();
    return 0;
}
