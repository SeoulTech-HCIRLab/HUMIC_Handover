cmake_minimum_required(VERSION 2.8.3)
project(humic_rl)

find_package(catkin REQUIRED COMPONENTS 
  rospy 
  std_msgs 
  trajectory_msgs 
  control_msgs 
  geometry_msgs 
  std_srvs 
  gazebo_msgs 
  cv_bridge
  sensor_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy std_msgs std_srvs trajectory_msgs control_msgs geometry_msgs  gazebo_msgs cv_bridge sensor_msgs
)

install(DIRECTORY launch nodes src
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

catkin_install_python(PROGRAMS
  nodes/humichandover_sac
  src/humic_handover_env.py
  src/handoverobject.py
  src/sac.py
  
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
