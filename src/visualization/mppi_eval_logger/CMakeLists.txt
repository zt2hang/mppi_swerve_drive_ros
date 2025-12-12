cmake_minimum_required(VERSION 3.0.2)
project(mppi_eval_logger)

find_package(catkin REQUIRED COMPONENTS
  rospy
  mppi_eval_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/mppi_eval_logger.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
