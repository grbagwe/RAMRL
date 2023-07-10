import os, sys
import random

import gym
from gym import Env
import numpy as np
from gym.spaces import Box
import pandas as pd
import shutil
import torch
import cv2
'''create a ramp and an ego vehicle using idm controller and check the rewards
'''
'''speed mode : 
default (alpl checks on) -> [0 1 1 1 1 1] -> Speed Mode = 31
most checks off (legacy) -> [0 0 0 0 0 0] -> Speed Mode = 0
all checks off -> [1 0 0 0 0 0] -> Speed Mode = 32
disable right of way check -> [1 1 0 1 1 1] -> Speed Mode = 55
run a red light [0 0 0 1 1 1] = 7 (also requires setSpeed or slowDown)
run a red light even if the intersection is occupied [1 0 0 1 1 1] = 39 (also requires setSpeed or slowDown)
'''
from PIL import Image

if 'SUMO_HOME' in os.environ:
    SUMO_HOME = os.environ['SUMO_HOME']
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    print('sumo_loaded')
tools = os.path.join(SUMO_HOME, 'tools')
sys.path.append(tools)
print(tools)
import traci as traci
import sumolib


class SumoRampEnv(Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(self, sumoConfPath=None):

        self.label = str(SumoRampEnv.CONNECTION_LABEL)
        self.CONNECTION_LABEL += 1
        self.alphasl0 = 0.7
        self.alphasl1 = 0.2
        self.rSuccess = 50
        self.alphaO = 0.03
        self.rTimeAlpha = 0.001
        self.alphaD = 0.001


        self.maxEpisodeLength = 3000  # 3600

        self.virtual_display = (1024, 1024)
        self.image_shape = (200, 768)  # 512,768
        # self.image_resize = (512, 512)

        self.episodeLength = self.maxEpisodeLength
        self.SUMO_HOME = self.isSUMOHOME()
        self.rC = -150
        self.maxSpeed = 55
        self.alphaJ = 0.5
        self.run = 0
        self.maxDistance = 400
        self.alphaDistancef = self.alphaDistancer = 0.02
        self.pastRlPosition =0  
        self.alphaP = 0.01

        # min and max acceleration
        self.observationDistance = 100  # observation distance for the rl vehicle

        self.min_acc = -1
        self.max_acc = +3
        self.oldAcc = self.min_acc
        obs_image_shape = self.image_shape
        # if self.image_resize in globals():
        #     obs_image_shape = self.image_resize

        self.action_space = Box(low=np.array([self.min_acc]), high=np.array([self.max_acc]), dtype=np.float32)
        spaces = {
            'image': Box(low=0, high=255,dtype=np.uint8, shape=(obs_image_shape[0], obs_image_shape[1], 3)),
            'velocity': Box(low=-100, high=100, shape=(7,)),
            'xPos': Box(low=-100000, high=100000, shape=(7,)),
            'yPos': Box(low=-100000, high=100000, shape=(7,)),
        }
        self.observation_space = gym.spaces.Dict(spaces)
        # self.observation_space = Box(low=)

        self.sumoConfigFile = "./custom_envs/sumo_ConfigTaper/ramp_2.sumocfg"
        # '/home/gauravb/Documents/MichiganTech/Programming/CustomRampTraining/custom_envs/sumo_Config/ramp_1.sumocfg'# sumoConfPath
        import traci as traci
        import sumolib
        self.sumoBinary = os.path.join(self.SUMO_HOME, "bin/sumo-gui")

    def reset(self):
        self.sumo_reset()
        self.done = False
        self.episodeLength = self.maxEpisodeLength
        self.info = {}
        self.rl_vehiclesSpawned = 0
        # print(get)
        return self.getObservations()

    def getobservedVehicles(self, vehicle_ids):

        self.rl_car_lane = traci.vehicle.getLaneID(self.rl_car_id)
        self.rl_car_pos = traci.vehicle.getPosition(self.rl_car_id)
        self.rl_car_distance = traci.vehicle.getDistance(self.rl_car_id)
        ObservedVehicleLane0 = []
        ObservedVehicleLane1 = []
        for vehicle in vehicle_ids:
            if vehicle.split('_')[0] != 'rl':
                vehiclePosition = traci.vehicle.getPosition(vehicle)
                vehicleDistance = traci.vehicle.getDistance(vehicle)
                VehicleRLDistance = abs(
                    vehicleDistance - self.rl_car_distance)  # abs distance of the vehicle the rl car
                vehicleLaneID = traci.vehicle.getLaneID(vehicle)

                if VehicleRLDistance < self.observationDistance:  # in observation
                    if vehicleLaneID.split('_')[1] == '0':  # in observation lane 0
                        # traci.vehicle.setColor(vehicle, color=(0, 255, 255, 255))
                        ObservedVehicleLane0.append((vehicle, VehicleRLDistance))
                    elif vehicleLaneID.split('_')[1] == '1':  # in observation lane 1
                        # traci.vehicle.setColor(vehicle, color=(0, 255, 255, 255))

                        ObservedVehicleLane1.append((vehicle, VehicleRLDistance))

            # vehicleSpeed = traci.vehicle.getSpeed(vehicle)
        ObservedVehicleLane0 = sorted(ObservedVehicleLane0, key=lambda ObservedVehicleLane0: ObservedVehicleLane0[1])[
                               :4]
        ObservedVehicleLane1 = sorted(ObservedVehicleLane1, key=lambda ObservedVehicleLane1: ObservedVehicleLane1[1])[
                               :2]
        return ObservedVehicleLane0, ObservedVehicleLane1

    def getLeadFollowercarBMerge(self, vehicle_ids):

        self.rl_car_lane = traci.vehicle.getLaneID(self.rl_car_id)
        self.rl_car_pos = traci.vehicle.getPosition(self.rl_car_id)
        self.rl_car_distance = traci.vehicle.getDrivingDistance(self.rl_car_id, "e35", 0.0)
        # print(self.rl_car_distance, 'rl car pos')
        leadVehicles = []
        rearVehicles = []
        for vehicle in vehicle_ids:
            if vehicle.split('_')[0] != 'bl':
                vehiclePosition = traci.vehicle.getPosition(vehicle)
                vehicleLane = traci.vehicle.getLaneID(vehicle)

                vehicleDistance = traci.vehicle.getDrivingDistance(vehicle, "e35", 0.0)
                distaneVehicleRl = vehicleDistance - self.rl_car_distance
                if vehicleLane.split('_')[1] == "0":
                    if distaneVehicleRl > 0:  # rear car
                        rearVehicles.append((vehicle, distaneVehicleRl))

                    elif distaneVehicleRl < 0:  ## lead car
                        leadVehicles.append((vehicle, distaneVehicleRl))

        if leadVehicles:
            leadVehicle = sorted(leadVehicles, key=lambda leadVehicles: leadVehicles[1])[-1]
        else:
            leadVehicle = ('no_vehicle', 400)
        if rearVehicles:
            rearVehicle = sorted(rearVehicles, key=lambda rearVehicles: rearVehicles[1])[-1]
        else:
            rearVehicle = ('no_vehicle', 0)
        return leadVehicle, rearVehicle

    def getObservations(self):
        # returns observations of the state

        state_speed = np.ones(7) * self.maxSpeed
        state_position_x = np.ones(7)
        state_position_y = np.ones(7)
        state_image = np.array(self.render())
        # print('state_image', state_image.shape)

        vehicle_ids = self.getVehicleIds()
        if vehicle_ids:
            obsLane0, obsLane1 = self.getobservedVehicles(vehicle_ids)

            for i, vehicle in enumerate(obsLane0):
                maxSpeed = traci.vehicle.getMaxSpeed(vehicle_ids[0])
                state_speed[i] = traci.vehicle.getSpeed(vehicle[0])
                state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
            for i, vehicle in enumerate(obsLane1, len(obsLane0)):
                state_speed[i] = traci.vehicle.getSpeed(vehicle[0])

                state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]

            # rl state information
            state_speed[-1] = traci.vehicle.getSpeed(self.rl_car_id) / self.maxSpeed
            state_position_x[-1] = traci.vehicle.getPosition(self.rl_car_id)[0]
            state_position_y[-1] = traci.vehicle.getPosition(self.rl_car_id)[1]
            #TODO set max and min values of clip from observation space
            state_speed = np.clip(state_speed, -100, 100)
            state_position_x = np.clip(state_position_x, -100000, 100000)
            state_position_y = np.clip(state_position_y, -100000, 100000)

        state = {'image': state_image.astype(np.uint8),
                 'xPos': np.array(state_position_x, dtype=np.float32),
                 'yPos': np.array(state_position_y, dtype=np.float32),
                 'velocity': np.array(state_speed, dtype=np.float32)}
        # self.obs_shape = len(state)
        # print('obs_shape',state[0].shape)
        return state

    def step(self, action):
        self.simstep()

        rl_car_acc_duration = traci.simulation.getDeltaT()
        rl_car_acc = action
        self.rl_car_id = self.addRLVehicle()
        currentSpeed = traci.vehicle.getSpeed(self.rl_car_id)
        next_vel = max([currentSpeed + rl_car_acc * rl_car_acc_duration,0])
        self.stepRltime += 1
        print(f"action {action}")
        print(traci.vehicle.getPosition(self.rl_car_id))
        print(f"next velocity: {next_vel}")


        traci.vehicle.slowDown(self.rl_car_id, next_vel, 1e-3)  # adjust acceleration of ego vehicle
        if traci.vehicle.getRoadID(self.rl_car_id) == "e23":
            if self.checkfirste23:
                self.info["mergeTime"]= self.time2merge
                self.checkfirste23 = False
        else:
            self.time2merge += traci.simulation.getDeltaT()

        # traci.vehicle.setAcceleration(self.rl_car_id, rl_car_acc, rl_car_acc_duration)
        self.episodeLength -= 1

        reward = self.calc_reward()

        state = self.getObservations()
        if self.isCollision():
            rl_collide = 1
        else:
            rl_collide = 0


        self.info["collision"]=rl_collide

        self.info["rlvehiclesSpawned"]= self.rl_vehiclesSpawned
        self.info["frame"] = state['image']
        if self.episodeLength == 0:
            self.done = True
            # self.close()
        else:
            self.done = False

        return state, reward, self.done, self.info

    def close(self):
        self.closeSim()

    def render(self, mode='rgb_array'):
        if self.virtual_display:
            # img = self.sumo.gui.screenshot(traci.gui.DEFAULT_VIEW,
            #                          f"temp/img{self.sim_step}.jpg",
            #                          width=self.virtual_display[0],
            #                          height=self.virtual_display[1])
            img = self.disp.grab()

            if mode == 'rgb_array':
                img = np.array(img)
                img = img[420:420 + self.image_shape[0], 0:0 + self.image_shape[1]]
                # if self.image_resize in globals():
                #     img = img.resize(self.image_resize)

                if not os.path.exists('./reshaped_img.png'):
                    cv2.imwrite('./reshaped_img.png', img)
                return np.array(img)
            return img

    def calc_reward(self):
        # return 1

        # calculate the reward
        vehicleAcc0 = []
        vehicleAcc1 = []
        rCongestion = 0
        rSuccess = 0

        reward = 0
        rTime = - abs(self.stepRltime) * self.rTimeAlpha
        
        # rlPosition = traci.vehicle.getDistance(self.rl_car_id)
        # #print(rlPosition, 'rlposition', self.alphaP, self.pastRlPosition)
        # rP = self.alphaP * np.sqrt(rlPosition - self.pastRlPosition)
        # self.pastRlPosition = rlPosition
        # rP = np.clip(rP, 1e-4,10)
        # rP = np.nan_to_num(rP)
        rP = 0

        print('rp', rP )
        if self.isCollision():
            rCollision = self.rC
        else:
            rCollision = 0

        # acc reward

        rlAcceleration = traci.vehicle.getAcceleration(self.rl_car_id)
        rlSpeed = traci.vehicle.getSpeed(self.rl_car_id)
        rlLane = traci.vehicle.getRoadID(self.rl_car_id)
        self.deltaT = traci.simulation.getDeltaT()

        if rlAcceleration > -200:  # avoid the max negative valye
            # if rlAcceleration<0:
            #     # print(rlAcceleration, 'rlAcceleration')

            maxAcc = traci.vehicle.getAccel(self.rl_car_id)
            rj = -1 * abs(self.alphaJ) * abs(rlAcceleration - self.oldAcc) / (maxAcc * self.deltaT)
            self.oldAcc = rlAcceleration
            # rs = +1 * abs(self.alphaJ) * rlSpeed / traci.vehicle.getMaxSpeed(self.rl_car_id)
        else:
            rj = 0

        if rlLane == 'e42':  # before merge

            leadVehicle, rearVehicle = self.getLeadFollowercarBMerge(self.getVehicleIds())
            leaddistance = leadVehicle[1]
            rearDistance = rearVehicle[1]

            rd = 2 - (abs(leaddistance) / self.maxDistance) ** self.alphaDistancef - (
                    rearDistance / self.maxDistance) ** self.alphaDistancer

            reward = rj + rd * self.alphaD + rTime + rP
            # print(reward, rj, rd, 'reward before merging')
            print(reward)
            return reward
        else:  # inmerge or post merge
            rearvehicle = traci.vehicle.getFollower(self.rl_car_id)
            leadvehicle = traci.vehicle.getLeader(self.rl_car_id)
            if rearvehicle and leadvehicle:
                rd = 2 - (abs(leadvehicle[1]) / self.maxDistance) ** self.alphaDistancef -  abs(
                            rearvehicle[1] / self.maxDistance) ** self.alphaDistancer
            else: rd= 0
            vehicle_ids = self.getVehicleIds()
            obsLane0, obsLane1 = self.getobservedVehicles(vehicle_ids)

            for i, vehicle in enumerate(obsLane0):
                if vehicle:
                    vehicleAcc0.append(traci.vehicle.getAcceleration(vehicle[0]))

            for i, vehicle in enumerate(obsLane1, len(obsLane0)):
                if vehicle:
                    vehicleAcc1.append(traci.vehicle.getAcceleration(vehicle[0]))
            # print(vehicleAcc0)
            if vehicleAcc0:
                meanAcc0 = np.mean(vehicleAcc0)
            else:
                meanAcc0 = 0
            if vehicleAcc1:
                 meanAcc1 = np.mean(vehicleAcc1)
            else:
                meanAcc1 = 0

            # congestion
            maxAcc = traci.vehicle.getAccel(self.rl_car_id)
            # print(' rcongestion', self.alphasl0 , meanAcc0 , maxAcc , self.alphasl1 , meanAcc1 , maxAcc)
            rCongestion = self.alphasl0 * meanAcc0 / maxAcc + self.alphasl1 * meanAcc1 / maxAcc

            steeringAngle = traci.vehicle.getAngle(self.rl_car_id)

            if steeringAngle > 0:  # to check if it has default value

                rO = - self.alphaO * (steeringAngle - self.previoussteeringAngle) / (self.deltaT)
                self.previoussteeringAngle = steeringAngle
                # TODO check 0.1 to variable simulation step
            else:
                rO = 0
            if rlLane == "e35" and self.checkSuccessOnce==0:
                self.checkSuccessOnce =1
            if self.checkSuccessOnce== 1:
                rSuccess = self.rSuccess
                self.checkSuccessOnce +=1

            else: rSuccess = 0

            reward = rj + rCongestion + rSuccess + rCollision + rO + self.alphaD * np.real(rd) + rTime + rP
            # print(reward, 'reward', rj,'rj', rCongestion, 'rCongestion', rSuccess, 'rsuc', rCollision, 'rCollision',
            #       rO, 'rO', rd, 'rd', rTime, 'rtime')

            # print(reward, 'reward final')
            print(reward)

            return reward

    def getStepCount(self):
        getTime = traci.simulation.getTime()
        return int(getTime)

    def sumo_reset(self):
        '''sumo_reset the env'''
        if self.run != 0:
            self.close()
            # self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        self.sumoBinary = os.path.join(self.SUMO_HOME, "bin/sumo-gui")
        sumoCmd = [self.sumoBinary,
                   "--start",
                   "--quit-on-end", "-c",
                   self.sumoConfigFile]
        sumoCmd.append('--random')
        # print(sumoCmd, 'sumocmd')

        #
        #
        # self.sumoBinary = os.path.join(self.SUMO_HOME, "bin/sumo-gui")
        # sumoCmd = [self.sumoBinary,
        #            "--start",
        #            "--quit-on-end", "-c", self.sumoConfigFile]
        if self.virtual_display is not None:
            sumoCmd.extend(['--window-size', f'{self.virtual_display[0]},{self.virtual_display[1]}'])
            from pyvirtualdisplay.smartdisplay import SmartDisplay
            print("Creating a virtual display.")
            self.disp = SmartDisplay(visible=0, size=self.virtual_display)
            self.disp.start()
            print("Virtual display started.")

        traci.start(sumoCmd, label='init_connection' + self.label)
        simstep = 0
        max_simsteps = 1000
        self.lastSavedFrame = 0
        self.rl_car_id = None

        traci.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")
        # self.step([+3])

    def isSUMOHOME(self):
        '''check if sumo is avaible'''

        if 'SUMO_HOME' in os.environ:
            self.SUMO_HOME = os.environ['SUMO_HOME']
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            print('sumo_loaded 2')
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")
        return self.SUMO_HOME

    def simstep(self):  # simulation Step
        '''simstep the simulation'''
        # self.saveImageFrame()
        traci.simulationStep()


    def isCollision(self):  # if collision occured in the simstep
        '''returns True if collision occured in the env'''

        isCollision = traci.simulation.getCollidingVehiclesNumber() > 0
        return isCollision

    def getVehicleIds(self):
        vehicle_ids = traci.vehicle.getIDList()
        return vehicle_ids

    def addRLVehicle(self):
        'checks id no rl car is present in vehicle then addsit'
        vehicle_ids = self.getVehicleIds()
        if self.rl_car_id not in vehicle_ids:
            self.rl_car_id = "rl_car_" + str(np.random.randint(1e10))  # random generate a rl_car

            traci.vehicle.add(self.rl_car_id, routeID="route_ramp")  # vehicle add
            traci.vehicle.setColor(self.rl_car_id, color=(255, 0, 0, 255))  # change vehicle color to red
            # print('type_id ', traci.vehicle.getTypeID(self.rl_car_id))
            traci.vehicle.setSpeedMode(self.rl_car_id, 32)  # change speed mode to 55 which is
            self.previoussteeringAngle = 0
            self.checkSuccessOnce = 0
            self.stepRltime = 0
            self.rl_vehiclesSpawned +=1
            self.time2merge = 0
            self.checkfirste23 = True
            self.pastRlPosition =0  
        return self.rl_car_id

    def addBaselineVehicle(self):
        'checks id no rl car is present in vehicle then addsit'
        vehicle_ids = self.getVehicleIds()
        if self.bl_car_id not in vehicle_ids:

            self.bl_car_id = "bl_car_" + str(np.random.randint(1e10))  # random generate a rl_car

            traci.vehicle.add(self.rl_car_id, routeID="route_ramp", typeID="idmAlternative", departSpeed = "random")  # vehicle add
            traci.vehicle.setColor(self.bl_car_id, color=(255 , 0, 0, 255))  # change vehicle color to blue
            # # print('type_id ', traci.vehicle.getTypeID(self.bl_car_id))
            traci.vehicle.setSpeedMode(self.bl_car_id, 55)  # change speed mode to 55 which is
            self.previoussteeringAngle = 0
            self.checkSuccessOnce = 0
            self.stepRltime = 0
        if len(vehicle_ids)!= 0:
            vehType = traci.vehicle.getTypeID(vehicle_ids[0])

        return self.bl_car_id

    def closeSim(self):
        traci.close()
        try:
            self.disp.stop()
        except AttributeError:
            pass


# simpath = "/home/gauravb/Documents/MichiganTech/Programming/CustomRampTraining/custom_envs/sumo_Config/ramp_1.sumocfg"
env = SumoRampEnv()
