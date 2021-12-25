#!/usr/bin/env python3
# -------------------------------
# Humic Handover Task Object
# Author: Kim Young Gi(HCIR Lab.)
# -------------------------------

""" ROS """
import rospy
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose

""" python Library """
import random
import time
import os
import numpy as np

class HandoverObject(object):
    def __init__(self, model):
        
        self.modelDir = model
        self.modelName = model
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.modelPath.replace('humic_rl/src', 'humic_gazebo/models/{0}/{1}.sdf'.format(self.modelDir, self.modelName))
       
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()

        self.object_position = Pose()

        self.init_check = True
                
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.check_model = False

        # self.last_index = 0
        # self.index = 0

    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == self.modelName:
                self.check_model = True

    def respawnModel(self):
        while True:
            if not self.check_model:
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.object_position, "world")
                rospy.loginfo("RespawnModel: %s", self.modelName)
                # rospy.loginfo("%s position: %.3f, %.3f, %.3f", self.modelName, self.object_position.position.x, self.object_position.position.y, self.object_position.position.z)
                break
            else:
                pass
    
    def deleteModel(self):
        while True:
            if self.check_model:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                del_model_prox(self.modelName)
                break
            else:
                pass

    def setPose(self, randompose=False, position_check=True):
        if randompose:
            # while position_check:
            #     goal_x_list = [ 0.67, 0.67, 0.67, 0.67,  0.67,  0.67, 0.67, 0.67, 0.67, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68,  0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69 ]
            #     goal_y_list = [ 0.0,  0.0,  0.0, -0.01, -0.01, -0.01, 0.01, 0.01, 0.01, 0.0,  0.0,  0.0, -0.01, -0.01, -0.01, 0.01, 0.01, 0.01, 0.0,  0.0,  0.0, -0.01, -0.01, -0.01, 0.01, 0.01, 0.01]
            #     goal_z_list = [1.10, 1.09,  1.1,  1.10,  1.09,   1.1, 1.10, 1.09, 1.1, 1.10, 1.09,  1.1,  1.10,  1.09,   1.1, 1.10, 1.09, 1.1, 1.10, 1.09,  1.1,  1.10,  1.09,   1.1, 1.10, 1.09, 1.1]
            #     self.index = random.randrange(0,len(goal_x_list))      
            
            #     if self.last_index == self.index:
            #         position_check = True
            #     else:
            #         self.last_index = self.index
            #         position_check = False

            # self.object_position.position.x = goal_x_list[self.index]
            # self.object_position.position.y = goal_y_list[self.index]
            # self.object_position.position.z = goal_z_list[self.index]

            self.object_position.position.x = random.uniform(0.67, 0.69)
            self.object_position.position.y = random.uniform(-0.01, 0.01)
            # self.object_position.position.y = 0.0
            self.object_position.position.z = random.uniform(1.09, 1.11)
        else:
            self.object_position.position.x = 0.68
            self.object_position.position.y = 0.0
            self.object_position.position.z = 1.10

        self.object_position.orientation.x = 0.0
        self.object_position.orientation.y = 0.0
        self.object_position.orientation.z = 0.0
        self.object_position.orientation.w = 1

        time.sleep(.5)
        self.respawnModel()

        return np.array([self.object_position.position.x, self.object_position.position.y, self.object_position.position.z], dtype=np.float)





