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

'''speed mode : 
default (all checks on) -> [0 1 1 1 1 1] -> Speed Mode = 31
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
        self.rSuccess = 100
        self.alphaO = 0.23
        self.rTimeAlpha = 0.8

        self.maxEpisodeLength = 3600  # 3600

        self.virtual_display = (1024, 1024)
        self.image_shape = (512, 768)  # 512,768

        self.episodeLength = self.maxEpisodeLength
        self.SUMO_HOME = self.isSUMOHOME()
        self.rC = -150
        self.maxSpeed = 55
        self.alphaJ = 0.5
        self.run = 0
        self.maxDistance = 400
        self.alphaDistancef = self.alphaDistancer = 0.05

        # min and max acceleration
        self.observationDistance = 100  # observation distance for the rl vehicle

        self.min_acc = -3
        self.max_acc = +3
        self.action_space = Box(low=np.array([self.min_acc]), high=np.array([self.max_acc]), dtype=np.float32)
        spaces = {
            'image': Box(low=0, high=255, shape=(self.image_shape[0], self.image_shape[1], 3)),
            'velocity': Box(low=-100, high=100, shape=(7,)),
            'xPos': Box(low=-100000, high=100000, shape=(7,)),
            'yPos': Box(low=-100000, high=100000, shape=(7,)),
        }
        self.observation_space = gym.spaces.Dict(spaces)
        # self.observation_space = Box(low=)

        self.sumoConfigFile = "./custom_envs/sumo_Config/ramp_1.sumocfg"
        # '/home/gauravb/Documents/MichiganTech/Programming/CustomRampTraining/custom_envs/sumo_Config/ramp_1.sumocfg'# sumoConfPath
        import traci as traci
        import sumolib
        self.sumoBinary = os.path.join(self.SUMO_HOME, "bin/sumo-gui")

    def reset(self):
        self.sumo_reset()
        self.done = False
        self.episodeLength = self.maxEpisodeLength
        self.info = {}
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
            if vehicle.split('_')[0] != 'rl':
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

        state = {'image': state_image.astype(float),
                 'xPos': state_position_x,
                 'yPos': state_position_y,
                 'velocity': state_speed}
        # self.obs_shape = len(state)
        # print('obs_shape',state[0].shape)
        return state

    def step(self, action):
        self.simstep()

        rl_car_acc_duration = 0.1
        rl_car_acc = 3
        self.rl_car_id = self.addRLVehicle()
        currentSpeed = traci.vehicle.getSpeed(self.rl_car_id)
        next_vel = max([currentSpeed + rl_car_acc * rl_car_acc_duration, 0])
        self.stepRltime += 1

        # self.__vehicles[vid]["accel"] = acc[i]
        #         this_vel = self.get_speed(vid)
        #         next_vel = max([this_vel + acc[i] * self.sim_step, 0])
        #         if smooth:
        #             self.kernel_api.vehicle.slowDown(vid, next_vel, 1e-3)

        traci.vehicle.slowDown(self.rl_car_id, next_vel, rl_car_acc_duration)  # adjust acceleration of ego vehicle

        # traci.vehicle.setAcceleration(self.rl_car_id, rl_car_acc, rl_car_acc_duration)

        self.episodeLength -= 1

        reward = self.calc_reward()

        state = self.getObservations()
        if self.isCollision():
            rl_collide = 1
            self.done = True
        else:
            rl_collide = 0


        self.info = {"collision":rl_collide}

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
                img = img[200:200 + self.image_shape[0], 0:0 + self.image_shape[1]]

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
        if self.isCollision():
            rCollision = self.rC
        else:
            rCollision = 0

        # acc reward

        rlAcceleration = traci.vehicle.getAcceleration(self.rl_car_id)
        rlSpeed = traci.vehicle.getSpeed(self.rl_car_id)
        rlLane = traci.vehicle.getRoadID(self.rl_car_id)
        if rlAcceleration > -200:  # avoid the max negative valye
            # if rlAcceleration<0:
            #     # print(rlAcceleration, 'rlAcceleration')

            maxAcc = traci.vehicle.getAccel(self.rl_car_id)
            rj = +1 * abs(self.alphaJ) * rlAcceleration / maxAcc
            # rs = +1 * abs(self.alphaJ) * rlSpeed / traci.vehicle.getMaxSpeed(self.rl_car_id)
        else:
            rj = 0

        if rlLane == 'e42':  # before merge

            leadVehicle, rearVehicle = self.getLeadFollowercarBMerge(self.getVehicleIds())
            leaddistance = leadVehicle[1]
            rearDistance = rearVehicle[1]

            rd = 2 - (abs(leaddistance) / self.maxDistance) ** self.alphaDistancef - (
                    rearDistance / self.maxDistance) ** self.alphaDistancer

            reward = rj + rd + rTime
            print(reward, rj, rd, 'reward before merging')
            return reward
        else:  # inmerge or post merge
            rearvehicle = traci.vehicle.getFollower(self.rl_car_id)
            leadvehicle = traci.vehicle.getLeader(self.rl_car_id)
            if rearvehicle and leadvehicle:
                rd = 2 - (abs(leadvehicle[1]) / self.maxDistance) ** self.alphaDistancef + (
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
                self.previoussteeringAngle = steeringAngle
                rO = - self.alphaO * (steeringAngle - self.previoussteeringAngle) / 0.1
                # TODO check 0.1 to variable simulation step
            else:
                rO = 0
            if rlLane == "e35" and self.checkSuccessOnce==0:
                self.checkSuccessOnce =1
            if self.checkSuccessOnce== 1:
                rSuccess = self.rSuccess
                self.checkSuccessOnce +=1

            else: rSuccess = 0

            reward = rj + rCongestion + rSuccess + rCollision + rO + np.real(rd) + rTime
            print('reward',rj, rCongestion, rSuccess, rCollision, rO , rd)

            # print(reward, 'reward final')

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

        self.rl_car_id = "0"
        # filepath = './results'
        #
        # # create files to save render image
        # if os.path.exists(filepath):
        #     shutil.rmtree(filepath)
        # if not os.path.exists(filepath):
        #     os.mkdir(filepath)
        #     self.imagePath = os.path.join(filepath, "simulation")
        #     os.mkdir(self.imagePath)
        traci.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

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

    # def saveImageFrame(self):
    #     '''caution when using this as this would be implemented after one simstep of simulation
    #     so the image is saved only after the simulationsimstep has occurred'''
    #     self.lastSavedFrame = self.getStepCount()
    #     imagePath = os.path.join(self.imagePath, 'frame_{}.png'.format(self.lastSavedFrame))
    #     traci.gui.screenshot(traci.gui.DEFAULT_VIEW, filename=imagePath)
    #
    # def getImageFrame(self, numFrames=1):
    #     frame = []
    #     if self.lastSavedFrame != 0:
    #         if numFrames == 1:  # return frame with one image if numf froames is one
    #             imagePath = os.path.join(self.imagePath, 'frame_{}.png'.format(self.lastSavedFrame))
    #             # open the image and convert it into array
    #             image = np.array(Image.open(imagePath))  # image is a 4 dimensional array with alpha channel
    #             image = image[:, :, :3]  # just use the rgb channel
    #             return image
    #
    #             # return Image.open(imagePath)
    #         elif numFrames > 1 and self.lastSavedFrame - numFrames >= 0:  # else return a stack of frame
    #             for i in range(0, numFrames):
    #                 imageFrame = self.lastSavedFrame - i
    #                 imagePath = os.path.join(self.imagePath, 'frame_{}.png'.format(self.lastSavedFrame))
    #                 image = np.array(Image.open(imagePath))
    #                 frame.append(image)
    #
    #     print('return frame')
    #     return frame

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
        # if vehicle_ids: # if vehicle id not empty
        #     route_sim = traci.route.getIDList() # route ids
        # print(vehicles_ids)
        if self.rl_car_id not in vehicle_ids:
            self.rl_car_id = "rl_car_" + str(np.random.randint(1e10))  # random generate a rl_car

            traci.vehicle.add(self.rl_car_id, routeID="route_ramp")  # vehicle add
            traci.vehicle.setColor(self.rl_car_id, color=(255, 0, 0, 255))  # change vehicle color to red
            # print('type_id ', traci.vehicle.getTypeID(self.rl_car_id))
            traci.vehicle.setSpeedMode(self.rl_car_id, 55)  # change speed mode to 55 which is
            self.previoussteeringAngle = 0
            self.checkSuccessOnce = 0
            self.stepRltime = 0
        return self.rl_car_id

    # def rlRandomAcc(self):
    #     if self.rl_car_id in self.getVehicleIds():
    #         rl_car_acc = 0 + np.double(np.random.randint(3))
    #         rl_car_acc_duration = 0.1
    #         traci.vehicle.setAcceleration(self.rl_car_id, rl_car_acc, rl_car_acc_duration)

    def closeSim(self):
        traci.close()
        try:
            self.disp.stop()
        except AttributeError:
            pass


# simpath = "/home/gauravb/Documents/MichiganTech/Programming/CustomRampTraining/custom_envs/sumo_Config/ramp_1.sumocfg"
env = SumoRampEnv()
#
# from matplotlib import pyplot as plt
# import cv2
#
# for episode in range(1, 10):
#     env.reset()
#     done = False
#     score = 0
#     while not done:
#         img = env.render()
#         # img = img[200:200+ self.image_shape[0], 0:0+ self.image_shape[1]]
#         # print('img shape', img.shape)
#         cv2.imwrite('./img_demo.png',img)
#         # cv2.imshow('img',img)
#         # cv2.waitKey()
#         # cv2.destroyAllWindows()
#         action = env.action_space.sample()
#         state_, reward, done, info = env.step(action)
#
#         score += reward
#     print(f'Episode {episode} score {score} ')

# number of simsteps to run
# import gym
from gym import Env
#
# simstep = 0
# max_simsteps = 1000
# while simstep<max_simsteps:
#
#     # print('simstep {}'.format(simstep))
#     rampEnv.saveImageFrame()
#     rampEnv.addRLVehicle()
#     rampEnv.simstep()
#     # rampEnv.rlRandomAcc()
#     if rampEnv.isCollision():
#         print('True')
#     simstep+=1


# get vehicle ids

# traci.vehicle.setColor(typeID= traci.vehicle.getTypeID(self.rl_car_id),color = (255,0,0,255))
# print('vehicle', vehicles_ids)


# screenshot of image
## remember the screenshot gets saved in the next simulation simstep. Hence add it first before opening the image.


# if vehicles_ids[0] == self.rl_car_id
#     traci.vehicle.setSpeed(self.rl_car_id,20)


# vehicle_speed = traci.vehicle.getSpeed(vehicles_ids[0])
# rl_car_speed = np.double(np.random.randint(30))

# print(self.rl_car_id,' speed ',traci.vehicle.getSpeed(self.rl_car_id), rl_car_acc , traci.vehicle.getAcceleration(self.rl_car_id), )
# print('self.rl_car_id',self.rl_car_id)

# print('vehicle id',vehicles_ids[0],vehicle_speed )
