# HCIR Lab(Human-Centered Intelligent Robotics Lab)
   * https://sites.google.com/view/hcir
## Humic Robot ROS Gazebo package

### Dependency
    $sudo apt-get install -y ros-noetic-joint-*
    $sudo apt-get install -y ros-noetic-gazebo-*
    $sudo apt-get install -y ros-noetic-joy ros-noetic-teleop-twist-joy ros-noetic-teleop-twist-keyboard 
    $sudo apt-get install -y ros-noetic-laser-proc ros-noetic-rgbd-launch ros-noetic-rosserial-arduino
    $sudo apt-get install -y ros-noetic-rosserial-python ros-noetic-rosserial-client ros-noetic-rosserial-msgs
    $sudo apt-get install -y ros-noetic-amcl ros-noetic-map-server ros-noetic-move-base ros-noetic-urdf ros-noetic-xacro
    $sudo apt-get install -y ros-noetic-compressed-image-transport ros-noetic-rqt-image-view ros-noetic-gmapping ros-noetic-navigation ros-noetic-interactive-markers
    
## How to run gazebo simulator
1. run launch file
      
        $ roslaunch humic_gazebo humic.launch

2. You can see Humic Robot in gazebo
  ![Humic_gazebo](https://user-images.githubusercontent.com/37207332/88047971-701d0a00-cb8d-11ea-8758-2aef9e656358.png)

3. You can control Joint
  * Control format: rostopic pub -1 /humic_control/${joint_name}/command std_msgs/Float64 "data: 0.0"
  * Joint name
         
         1. Right Arm: r_shoulder, r_upperArm, r_elbow, r_foreArm, r_lowerArm, r_wrist
         2. Left Arm: l_shoulder, l_upperArm, l_elbow, l_foreArm, l_lowerArm, l_wrist
         3. Column
         4. neck
         5. head
         6. Right Finger: r_finger1_1, r_finger1_2, r_finger2_1, r_finger2_2, r_finger3_1, r_finger3_2, r_finger4_1, r_finger4_2, r_thumb
         7. Left Finger: l_finger1_1, l_finger1_2, l_finger2_1, l_finger2_2, l_finger3_1, l_finger3_2, l_finger4_1, l_finger4_2, l_thumb
