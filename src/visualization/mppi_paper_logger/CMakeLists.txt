cmake_minimum_required(VERSION 3.0.2)
project(mppi_paper_logger)

find_package(catkin REQUIRED COMPONENTS
  rospy
  nav_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/mppi_paper_logger.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
