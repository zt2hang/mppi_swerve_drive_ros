cmake_minimum_required(VERSION 3.0.2)
project(mppi_ilc_prior)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(USE_OPENMP "Enable OpenMP for parallel sampling" ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  visualization_msgs
  nav_msgs
  std_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  grid_map_core
  grid_map_ros
  grid_map_msgs
  jsk_rviz_plugins
  mppi_eval_msgs
  mppi_hc
)

find_package(Boost REQUIRED COMPONENTS filesystem system)

find_package(Eigen3 REQUIRED)

if(USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_FOUND)
    message(STATUS "[MPPI-ILC-PRIOR] OpenMP found. Parallel sampling enabled.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  else()
    message(WARNING "[MPPI-ILC-PRIOR] OpenMP not found. Using single-threaded sampling.")
  endif()
endif()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES mppi_ilc_prior_core
  CATKIN_DEPENDS
    roscpp
    geometry_msgs
    visualization_msgs
    nav_msgs
    std_msgs
    tf2
    tf2_geometry_msgs
    tf2_ros
    grid_map_core
    grid_map_ros
    grid_map_msgs
    jsk_rviz_plugins
    mppi_eval_msgs
    mppi_hc
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
)

add_library(mppi_ilc_prior_core
  src/mppi_ilc_prior_core.cpp
)
target_link_libraries(mppi_ilc_prior_core
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
)
add_dependencies(mppi_ilc_prior_core ${catkin_EXPORTED_TARGETS})

add_executable(mppi_ilc_prior_node
  src/mppi_ilc_prior_node.cpp
  src/mppi_ilc_prior_ros.cpp
)
target_link_libraries(mppi_ilc_prior_node
  mppi_ilc_prior_core
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
)
add_dependencies(mppi_ilc_prior_node ${catkin_EXPORTED_TARGETS})

install(TARGETS mppi_ilc_prior_core mppi_ilc_prior_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
