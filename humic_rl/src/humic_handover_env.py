#!/usr/bin/env python3
# HUMIC Handover Environment
# HCIR Lab.(Young-Gi Kim)

import rospy
import rospkg
from std_srvs.srv import Empty
from gazebo_msgs.msg import ContactsState
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float64

import time
import numpy as np
from numpy.linalg import norm

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import math

import torch
import torchvision.models as models
import torchvision.transforms as transforms

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()

from PyKDL import *
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import *

from src.handoverobject import HandoverObject

class MSG_INVALID_JOINT_NAMES_DIFFER(Exception):
    """ Error object exclusively raised by _process_observations. """
    pass

class HumicHandoverGazeboEnv(object):
    def __init__(self):

        with open(os.path.join(rospkg.RosPack().get_path('humic_description'), 'urdf/humic.urdf.xacro'), 'r') as f:
            robot = URDF.from_xml_string(f.read())

        self.right_arm_kdl = KDLKinematics(robot, 'base_footprint', 'r_hand_link')
        
        self.left_arm_kdl = KDLKinematics(robot, 'base_footprint', 'l_hand_link')
        
        """ Right Arm """
        self.right_arm_joints = (
            'r_shoulder_joint', 
            'r_upperArm_joint', 
            'r_elbow_joint', 
            'r_foreArm_joint', 
            'r_lowerArm_joint', 
            'r_wrist_joint'
        )

        self.right_arm_pub = rospy.Publisher('/humic/right_arm/command', JointTrajectory, queue_size=1)

        """ Left Arm """
        self.left_arm_joints = (
            'l_shoulder_joint', 
            'l_upperArm_joint', 
            'l_elbow_joint', 
            'l_foreArm_joint', 
            'l_lowerArm_joint', 
            'l_wrist_joint'
        )

        self.left_arm_pub = rospy.Publisher('/humic/left_arm/command', JointTrajectory, queue_size=1)

        """ Image Feature networks models """        
        # # resnet18 model
        self.model = models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.model.to(device)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        """ State & Action """
        self.observation_dim = 512 + 14 # depth_image(1x84x84, resized), joint angles(8), ee-position(6)
        
        self.fullstate_dim = 8 + 6 + 6 + 2 + 3 # joint angles(8), ee-positions(6), target positions(6), distances(2), box_size(3)

        self.action_dim = 8 # joint angles
        
        action_shape = (self.action_dim, )
        
        low =  np.array([   
            math.radians(0),  
            math.radians(0), 
            -math.radians(30), 
            -math.radians(20), 
            -math.radians(72),  
            math.radians(0), 
            -math.radians(30), 
            -math.radians(20)], dtype=np.float) # 72 degree, 75, 30, 20
        
        high = np.array([ 
            math.radians(72), 
            math.radians(75),  
            math.radians(30), 
            math.radians(0),  
            math.radians(0), 
            math.radians(75),  
            math.radians(30),
            math.radians(0)], dtype=np.float)
        
        action_scale = np.array([
            math.radians(72), 
            math.radians(75), 
            math.radians(30), 
            math.radians(20), 
            math.radians(72), 
            math.radians(75), 
            math.radians(30), 
            math.radians(20)], dtype=np.float)
        
        action_bias = np.array([
            math.radians(72), 
            math.radians(75), 
            math.radians(0), 
            -math.radians(20), 
            -math.radians(72), 
            math.radians(75), 
            math.radians(0), 
            -math.radians(20)], dtype=np.float)
        
        action_div = np.array([2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0], dtype=np.float)

        self.action_space = {
                                'low'   : low,
                                'high'  : high,
                                'action_scale': action_scale,
                                'action_bias': action_bias,
                                'action_div': action_div,
                                'shape' : action_shape
        }

        self.init_action = np.zeros(12)
        # self.right_init_action = np.array([1.57, 0, 0, 0, 0, 0])
        # self.left_init_action = np.array([-1.57, 0, 0, 0, 0, 0])

        """ Gazebo """
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)

        """ Handover Objects """
        self.dirPath = rospkg.RosPack().get_path('humic_rl')

        self.objects = (
                            ('human_box_10x20x20', np.array([0, -0.1, 0], dtype=np.float), np.array([0, 0.1, 0], dtype=np.float), np.array([0.1, 0.2, 0.2], dtype=np.float)), # object, right point, left point, size
        )

        # self.objects = (
        #                     ('human_box_15x20x20', np.array([0, -0.1, 0], dtype=np.float), np.array([0, 0.1, 0], dtype=np.float), np.array([0.1, 0.2, 0.2], dtype=np.float)), # object, right point, left point, size
        #                     ('human_box_15x15x20', np.array([0, -0.075, 0], dtype=np.float), np.array([0, 0.075, 0], dtype=np.float), np.array([0.1, 0.15, 0.2], dtype=np.float)), # object, right point, left point, size
        #                     ('human_box_15x10x20', np.array([0, -0.05, 0], dtype=np.float), np.array([0, 0.05, 0], dtype=np.float), np.array([0.1, 0.1, 0.2], dtype=np.float)), # object, right point, left point, size                            
        # )
                          
        self.init_object = True
        self.objects_num = len(self.objects)
        self.last_query_index = self.query_idx = 0
        
        """ Reward """
        self.reward_success = 2
        self.reward_contact = 2
        self.reward_collision = -1

        self.reward_pub = rospy.Publisher('/humic/reward', Float64, queue_size=1)
        self.dist_HRTR_pub = rospy.Publisher('/humic/distHRTR', Float64, queue_size=1)
        self.dist_HLTL_pub = rospy.Publisher('/humic/distHLTL', Float64, queue_size=1)

        self.success = False

        self.max_move = 0.1

        self.dist_goal = 0.025 # ball contact, < 1.0cm
        self.dist_contact = 0.025

        """ Collision """
        self.collision_sub = rospy.Subscriber('/humic/gazebo_contact', ContactsState, self.collisionCallback)
        self.collision_msg = None
        
        self.r_collision = None
        self.l_collision = None

        self.dont_collide = 1
        self.can_collide= 2

    """ ROS Message Check """
    def checkJointsStateMsg(self, msg, joint_names):
        if not msg:
            print("Message is empty")
        else:
            if msg.joint_names != joint_names:
                if len(msg.joint_names) != len(joint_names):
                    raise MSG_INVALID_JOINT_NAMES_DIFFER
        
        return np.array(msg.actual.positions, dtype=float)
    
    """ Convert data to ROS Message """
    def getTrajectoryMsg(self, joint_names, angles):
        arm_msg = JointTrajectory()
        arm_msg.joint_names = joint_names
        arm_msg.header.stamp = rospy.Time.now()

        arm_target = JointTrajectoryPoint()
        arm_target.positions = angles
        arm_target.time_from_start.secs = 1

        arm_msg.points = [arm_target]
        return arm_msg
    
    def collisionCallback(self, msg):
        self.collision_msg = msg
        if self.collision_msg.states:
            if 'r_' in self.collision_msg.states[0].collision1_name and 'humic' in self.collision_msg.states[0].collision2_name:
                self.r_collision = self.dont_collide
            elif 'r_' in self.collision_msg.states[0].collision1_name and 'object' in self.collision_msg.states[0].collision2_name:
                self.r_collision = self.can_collide
            elif 'r_' in self.collision_msg.states[0].collision1_name and 'human' in self.collision_msg.states[0].collision2_name:
                self.r_collision = self.dont_collide
            
            if 'l_' in self.collision_msg.states[0].collision1_name and 'humic' in self.collision_msg.states[0].collision2_name:
                self.l_collision = self.dont_collide
            elif 'l_' in self.collision_msg.states[0].collision1_name and 'object' in self.collision_msg.states[0].collision2_name:
                self.l_collision = self.can_collide
            elif 'l_' in self.collision_msg.states[0].collision1_name and 'human' in self.collision_msg.states[0].collision2_name:
                self.l_collision = self.dont_collide

        self.collision_msg = None

    # ------------------------------------------------------------------------------------------------------------------------------- #
    def action_sample(self):
        return np.random.uniform(low=self.action_space['low'], high=self.action_space['high'], size=self.action_space['shape'])
    
    def step(self, action):
        action = np.clip(action, self.action_space['low'], self.action_space['high'])

        right_action = np.concatenate([action[:3], 0, action[3], 0], axis=None)
        left_action  = np.concatenate([action[4:7], 0, action[7], 0], axis=None)

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_proxy()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")
        
        """ Action """
        self.right_arm_pub.publish(self.getTrajectoryMsg(self.right_arm_joints, right_action))
        self.left_arm_pub.publish(self.getTrajectoryMsg(self.left_arm_joints, left_action))

        """" Obsevation"""
        right_joints_state_msg = None
        while right_joints_state_msg is None:
            try:
                right_joints_state_msg = rospy.wait_for_message('/humic/right_arm/state', JointTrajectoryControllerState, timeout=5)
            except:
                pass
        
        left_joints_state_msg = None
        while left_joints_state_msg is None:
            try:
                left_joints_state_msg = rospy.wait_for_message('/humic/left_arm/state', JointTrajectoryControllerState, timeout=5)
            except:
                pass
        
        # Observation Image
        obs_rgb_img_state_msg = None
        while obs_rgb_img_state_msg is None:
            try:
                obs_rgb_img_state_msg = rospy.wait_for_message('/kinect2/hd/image_color_rect', Image, timeout=5)
            except:
                pass
        
        # Collision Check
        r_collision = self.r_collision
        l_collision = self.l_collision
        
        self.r_collision = None
        self.l_collision = None
        self.collision_msg = None
        
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_proxy()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        last_right_joints_state = self.checkJointsStateMsg(right_joints_state_msg, self.right_arm_joints) # get joint angles from msg      

        last_left_joints_state = self.checkJointsStateMsg(left_joints_state_msg, self.left_arm_joints)

        right_pos, _ = self.right_arm_kdl.FK(last_right_joints_state)
        right_pos = np.ravel(right_pos, order='C')

        left_pos, _ = self.left_arm_kdl.FK(last_left_joints_state)
        left_pos = np.ravel(left_pos, order='C')

        # Distance between right hand pos and query object pos / Distance between left hand pos and query object pos / L2Norm
        cur_distHRTR = norm(self.query_object_right_target - right_pos)
        cur_distHLTL = norm(self.query_object_left_target - left_pos)
        
        # RGB Image State
        obs_rgb_img_state_msg = CvBridge().imgmsg_to_cv2(obs_rgb_img_state_msg, 'bgr8')
        obs_rgb_img_state = self.transform(obs_rgb_img_state_msg).to(device)
        img_state = self.model(obs_rgb_img_state.unsqueeze(0)).squeeze()
        
        """ Get Reward """   
        done = False

        diff_HRTR = abs(cur_distHRTR - self.pre_distHRTR)
        diff_HLTL = abs(cur_distHLTL - self.pre_distHLTL)

        reward = (
            -math.log(cur_distHRTR + 10e-8)
            -math.log(cur_distHLTL + 10e-8)
            -cur_distHRTR
            -cur_distHLTL
            +(np.where(diff_HRTR <= self.max_move, 0, -(diff_HRTR/(cur_distHRTR + 1))))# distance difference penalty
            +(np.where(diff_HLTL <= self.max_move, 0, -(diff_HLTL/(cur_distHLTL + 1))))# distance difference penalty
        )
        
        self.pre_distHRTR = cur_distHRTR
        self.pre_distHLTL = cur_distHLTL
            
        # Collide reward
        if r_collision is not None:
            done = True
            if r_collision == self.dont_collide:
                reward += self.reward_collision
                print("Right Collision Humic or Human")
            
        if l_collision is not None:
            done = True
            if l_collision == self.dont_collide:
                reward += self.reward_collision
                print("Left Collision Humic or Human")
        
        # Get Goal, Holding
        if cur_distHRTR < self.dist_contact:
            # print("Right Contact")
            reward += self.reward_contact

        if cur_distHLTL < self.dist_contact:
            # print("Left Contact")
            reward += self.reward_contact

        if cur_distHRTR < self.dist_goal and cur_distHLTL < self.dist_goal: # success
            print("Right and Left contact, Right dist:{}\tLeft dist:{}".format(cur_distHRTR, cur_distHLTL))
            reward += self.reward_success
            self.success = True
            done = True
        
        # Reward Scaling
        reward *= 0.1
        
        """ Get State """
        full_state = np.r_[
                    np.reshape(np.concatenate((last_right_joints_state[:3], last_right_joints_state[4]), axis=None), -1), # 4
                    np.reshape(np.concatenate((last_left_joints_state[:3], last_left_joints_state[4]), axis=None), -1), # 4
                    np.reshape(right_pos, -1), # 3
                    np.reshape(left_pos, -1), # 3
                    np.reshape(self.query_object_right_target, -1), # 3
                    np.reshape(self.query_object_left_target, -1), # 3
                    np.reshape(cur_distHRTR, -1), #1
                    np.reshape(cur_distHLTL, -1), #1
                    np.reshape(self.query_object_size, -1)  # 3
        ]
        
        observation_state = np.r_[
                    np.reshape(img_state.cpu().clone().detach().numpy(), -1), 
                    np.reshape(np.concatenate((last_right_joints_state[:3], last_right_joints_state[4]), axis=None), -1), # 4
                    np.reshape(np.concatenate((last_left_joints_state[:3], last_left_joints_state[4]), axis=None), -1), # 4
                    np.reshape(right_pos, -1), # 3
                    np.reshape(left_pos, -1) # 3
        ]

        self.reward_pub.publish(reward)
        self.dist_HRTR_pub.publish(cur_distHRTR)
        self.dist_HLTL_pub.publish(cur_distHLTL)

        state = {'observation': observation_state, 'fullstate':full_state}

        return state, reward, done, self.success

    def reset(self, evaluate=False, change_obj_pose=False):
        # # Right Arm
        right_arm_init_angles = self.getTrajectoryMsg(self.right_arm_joints, self.init_action[:6])
        # right_arm_init_angles = self.getTrajectoryMsg(self.right_arm_joints, self.right_init_action)
        # Left Arm
        left_arm_init_angles = self.getTrajectoryMsg(self.left_arm_joints, self.init_action[6:])
        # left_arm_init_angles = self.getTrajectoryMsg(self.left_arm_joints, self.left_init_action)

        self.right_arm_pub.publish(right_arm_init_angles)
        self.left_arm_pub.publish(left_arm_init_angles)

        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_world_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_world service call failed")
        
        time.sleep(0.2)
        
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_proxy()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        if evaluate:
            if self.init_object:
                self.init_object = False
                self.getQueryObject(query_object_change=False, randompose=True)
            elif self.success:
                self.query_object.deleteModel()
                self.getQueryObject(query_object_change=False, randompose=True)
                self.success = False
            elif change_obj_pose:
                self.query_object.deleteModel()
                self.getQueryObject(query_object_change=False, randompose=True)
        else:
            if self.init_object:
                self.init_object = False
                self.getQueryObject(query_object_change=False, randompose=False)
                # self.getQueryObject(query_object_change=False, randompose=True) # Re-start
            elif self.success:
                self.query_object.deleteModel()
                self.getQueryObject(query_object_change=False, randompose=True)
                self.success = False
            elif change_obj_pose:
                self.query_object.deleteModel()
                self.getQueryObject(query_object_change=False, randompose=True)
        
        """" Obsevation"""
        right_joints_state_msg = None
        while right_joints_state_msg is None:
            try:
                right_joints_state_msg = rospy.wait_for_message('/humic/right_arm/state', JointTrajectoryControllerState, timeout=5)
            except:
                pass
        
        left_joints_state_msg = None
        while left_joints_state_msg is None:
            try:
                left_joints_state_msg = rospy.wait_for_message('/humic/left_arm/state', JointTrajectoryControllerState, timeout=5)
            except:
                pass
        
        # Observation Image
        obs_rgb_img_state_msg = None
        while obs_rgb_img_state_msg is None:
            try:
                obs_rgb_img_state_msg = rospy.wait_for_message('/kinect2/hd/image_color_rect', Image, timeout=5)
            except:
                pass
        
        self.r_collision = None
        self.l_collision = None
        self.collision_msg = None
        
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_proxy()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        
        # Joint State, FK
        last_right_joints_state = self.checkJointsStateMsg(right_joints_state_msg, self.right_arm_joints) # get joint angles from msg 
        
        last_left_joints_state = self.checkJointsStateMsg(left_joints_state_msg, self.left_arm_joints)

        right_pos, _ = self.right_arm_kdl.FK(last_right_joints_state)
        right_pos = np.ravel(right_pos, order='C')

        left_pos, _ = self.left_arm_kdl.FK(last_left_joints_state)
        left_pos = np.ravel(left_pos, order='C')

        # Distance between right hand pos and query object pos / Distance between left hand pos and query object pos / L2Norm
        self.pre_distHRTR = norm(self.query_object_right_target - right_pos)
        self.pre_distHLTL = norm(self.query_object_left_target - left_pos)
        
        # RGB Image State
        obs_rgb_img_state_msg = CvBridge().imgmsg_to_cv2(obs_rgb_img_state_msg, 'bgr8')
        obs_rgb_img_state = self.transform(obs_rgb_img_state_msg).to(device)
        img_state = self.model(obs_rgb_img_state.unsqueeze(0)).squeeze()
            
        """ Get State """
        full_state = np.r_[
                    np.reshape(np.concatenate((last_right_joints_state[:3], last_right_joints_state[4]), axis=None), -1), # 4
                    np.reshape(np.concatenate((last_left_joints_state[:3], last_left_joints_state[4]), axis=None), -1), # 4
                    np.reshape(right_pos, -1), # 3
                    np.reshape(left_pos, -1), # 3
                    np.reshape(self.query_object_right_target, -1),
                    np.reshape(self.query_object_left_target, -1),
                    np.reshape(self.pre_distHRTR, -1), #1
                    np.reshape(self.pre_distHLTL, -1), #1
                    np.reshape(self.query_object_size, -1) # 3
        ]

        observation_state = np.r_[
                    np.reshape(img_state.cpu().clone().detach().numpy(), -1), 
                    np.reshape(np.concatenate((last_right_joints_state[:3], last_right_joints_state[4]), axis=None), -1), # 4
                    np.reshape(np.concatenate((last_left_joints_state[:3], last_left_joints_state[4]), axis=None), -1), # 4
                    np.reshape(right_pos, -1), # 3
                    np.reshape(left_pos, -1) # 3
        ]

        state = {'observation': observation_state, 'fullstate':full_state}

        return state

    def getQueryObject(self, query_object_change=False, randompose=False):
        if query_object_change: # changing query object 
            self.query_idx = (self.query_idx + 1) % self.objects_num
            self.last_query_index = self.query_idx
                
        query_object = self.objects[self.query_idx][0]
        self.query_object = HandoverObject(model=query_object)
        query_object_pos = self.query_object.setPose(randompose)
        print("Objects Position: xyz:{}".format(query_object_pos))
        self.query_object_right_target = query_object_pos + self.objects[self.query_idx][1]
        self.query_object_left_target = query_object_pos + self.objects[self.query_idx][2]
        self.query_object_size = self.objects[self.query_idx][3]
        time.sleep(3)
    
class HumicHandoverRealWorldEnv(object):
    def __init__(self):
        pass