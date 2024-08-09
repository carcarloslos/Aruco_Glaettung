cmake_minimum_required(VERSION 3.0.2)
project(aruco_filter)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)

find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp std_msgs
  DEPENDS Eigen3
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/include/aruco_glaetter
)

add_executable(signal_filter src/signal_filter.cpp)

target_link_libraries(signal_filter
  ${catkin_LIBRARIES}
)

add_dependencies(signal_filter ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
