cmake_minimum_required(VERSION 2.8.3)
project(humic_gazebo)

find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
  controller_manager
  joint_state_controller
  robot_state_publisher
)

catkin_package(
  CATKIN_DEPENDS gazebo_ros gazebo_msgs
  DEPENDS gazebo
)

link_directories(${GAZEBO_LIBRARY_DIRS})

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(DIRECTORY launch worlds config models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
