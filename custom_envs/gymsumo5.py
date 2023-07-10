import os, sys
import random



# " updated env from Lucas sumo_rl"
# @misc{sumorl,
#     author = {Lucas N. Alegre},
#     title = {{SUMO-RL}},
#     year = {2019},
#     publisher = {GitHub},
#     journal = {GitHub repository},
#     howpublished = {\url{https://github.com/LucasAlegre/sumo-rl}},
# }


import gym
from gym import Env
import numpy as np
from gym.spaces import Box
import pandas as pd
import shutil
import torch
import cv2
from typing import Callable, Optional, Tuple, Union
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
LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ


class SumoRamp(gym.Env):
    metadata = {'render.modes': ['human','rgb_array']}
    CONNECTION_LABEL = 0

    def __init__(self, weights=None, action_space={"high": 3, "low": -4.5},
            sumoParameters=None, isBaseline=False, obsspaces=None, render=0,sumo_seed: Union[str,int] = 'random',sumo_warnings: bool = True,
                 net_file: str = './custom_envs/sumo_ConfigTaper/Ramp_2.net.xml',
                 route_file :str = './custom_envs/sumo_ConfigTaper/ramp_2.rou.xml'):
        print('weights this ', weights)

        print(self._sumo_binary, '\n\n\n sumo binary')
        self.alphasl0 = weights['alphasl0']  # 0.7
        self.alphasl1 = weights['alphasl1']  # 0.2
        self.rSuccess = weights['rSuccess']  # 50
        self.alphaO = weights['alphaO']  # 0.03
        self.rTimeAlpha = weights['rTimeAlpha']  # 0.001
        self.alphaD = weights['alphaD']  # 0.001

        self.sumo_seed = sumo_seed
        self.sumo_warnings = sumo_warnings
        self.use_gui = True
        self._net = net_file
        self._route = route_file
        
        self.maxEpisodeLength = sumoParameters['episodeLength']  # 3000  # 3600

        self.virtual_display = (1024, 1024)
        self.image_shape = (200, 768)  # 512,768
        # self.image_resize = (512, 512)

        self.episodeLength = self.maxEpisodeLength
        self.env_render = render
       
        if self.use_gui:
            self._sumo_binary= sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary= sumolib.checkBinary('sumo')
        self.label = str(SumoRamp.CONNECTION_LABEL)
        SumoRamp.CONNECTION_LABEL += 1
        self.sumo = None
        #self.SUMO_HOME = self.isSUMOHOME()
        if LIBSUMO:
            traci.start([sumolib.checkBinary('sumo'), '-n', self._net])  # Start only to retrieve traffic light information
            conn = traci
        else:
            traci.start([sumolib.checkBinary('sumo'), '-n', self._net], label='init_connection'+self.label)
            conn = traci.getConnection('init_connection'+self.label)
        conn.close()



        self.rC = weights['rC']  # -150
        self.maxSpeed = 30  # traci.vehicle.getMaxSpeed#30
        self.alphaJ = weights['alphaJ']  # 0.5
        self.run = 0
        self.maxDistance = 400  # weights['maxDistance']  # 400
        self.alphaDistancef = self.alphaDistancer = weights['alphaDistance']  # .02
        self.pastRlPosition = 0
        self.alphaP = weights['alphaP']  # 0.01

        # min and max acceleration
        self.observationDistance = 100  # observation distance for the rl vehicle

        self.min_acc = -1 * abs(action_space['low'])
        self.max_acc = action_space['high']
        self.oldAcc = self.min_acc
        obs_image_shape = self.image_shape
        self.isBaseline = isBaseline
        # if self.image_resize in globals():
        #     obs_image_shape = self.image_resize

        self.action_space = Box(low=np.array([self.min_acc]), high=np.array([self.max_acc]), dtype=np.float32)
        # obsspaces = {
        #     'image': Box(low=0, high=255, dtype=np.uint8, shape=(obs_image_shape[0], obs_image_shape[1], 3)),
        #     'velocity': Box(low=0, high=70, shape=(7,)),
        #     'xPos': Box(low=-100000, high=100000, shape=(7,)),
        #     'yPos': Box(low=-100000, high=100000, shape=(7,)),
        # }
        self.observation_space = gym.spaces.Dict(obsspaces)
        # self.observation_space = Box(low=)
        
        self.sumoConfigFile = "./custom_envs/sumo_ConfigTaper/ramp_2.sumocfg"

        self.sumoBinary = os.path.join(self.SUMO_HOME, "bin/sumo-gui")



    def reset(self):
        self.sumo_reset()
        self.done = False
        self.episodeLength = self.maxEpisodeLength
        self.info = {}
        self.rl_vehiclesSpawned = 0
        #for _ in range(20):
        #   self.simstep()
        return self.getObservations()

    def sumo_reset(self):
        '''sumo_reset the env'''
        if self.run != 0:
            self.close()
        self.run += 1
        simstep = 0
        max_simsteps = 1000
        self.lastSavedFrame = 0
        self.rl_car_id = None
        # self._sumo_binary = self.sumoBinary

        sumo_cmd = [self._sumo_binary,
                     '-n', './custom_envs/sumo_ConfigTaper/Ramp_2.net.xml',
                     '-r', './custom_envs/sumo_ConfigTaper/ramp_2.rou.xml',
                     '--waiting-time-memory', '10000',
                     '--time-to-teleport', '-1', '--random']
        if self.use_gui:
            sumo_cmd.extend(['--start', '--quit-on-end'])
            if self.virtual_display is not None:
                sumo_cmd.extend(['--window-size', f'{self.virtual_display[0]},{self.virtual_display[1]}'])
                from pyvirtualdisplay.smartdisplay import SmartDisplay
                print("Creating a virtual display.")
                
                self.disp = SmartDisplay(size=self.virtual_display)
                print(self.virtual_display)
                self.disp.start()
                print("Virtual display started.")

        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)
        
        if self.use_gui:

            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")                
        print(self.sumo, ' self.sumo')


        #for _ in range(20):
        #   self.simstep()


    def isSUMOHOME(self):
        '''check if sumo is avaible'''

        if 'SUMO_HOME' in os.environ:
            self.SUMO_HOME = os.environ['SUMO_HOME']
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            print('sumo_loaded ')
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")
        return self.SUMO_HOME

    def simstep(self):  # simulation Step
        '''simstep the simulation'''
        traci.simulationStep()

    def render(self, mode='rgb_array'):
        if self.virtual_display:
            img = self.disp.grab()
            if mode == 'rgb_array':
                img = np.array(img)
                img = img[420:420 + self.image_shape[0], 0:0 + self.image_shape[1]]
                #cv2.imwrite(f"./sumo_images/reshaped_img_{traci.simulation.getTime}.png", img)
                #print(f"./sumo_images/reshaped_img_{traci.simulation.getTime}.png")
                return np.array(img)
            return img

    def getobservedVehicles(self, vehicle_ids):
        
        #self.rl_car_lane = traci.vehicle.getLaneID(self.rl_car_id)
        #self.rl_car_pos = traci.vehicle.getPosition(self.rl_car_id)
        try :
            self.rl_car_distance = traci.vehicle.getDistance(self.rl_car_id)
        except:
            print('exception rl not found occured',vehicle_ids)
            return [], []
        ObservedVehicleLane0 = []
        ObservedVehicleLane1 = []
        #print('rl_lane ',traci.vehicle.getRoadID(self.rl_car_id))


        leadVehicle, rearVehicle = self.getLeadFollowercarBMerge(vehicle_ids)
        #print(traci.vehicle.getDistance(self.rl_car_id))

        if "no_vehicle" in leadVehicle:
                leadVehicle2 =('no_vehicle',400)
        else:
            leadVehicle2 = traci.vehicle.getLeader(leadVehicle[0],100)
            if leadVehicle2:

                if set(leadVehicle2) & set(["",None]):
                    leadVehicle2 = ('no_vehicle',400)
                else:
                    traci.vehicle.setColor(leadVehicle2[0], color=(255, 165,165 , 255))


        if "no_vehicle"  in rearVehicle:
            rearVehicle2 =('no_vehicle',0)
        else:
            rearVehicle2 =traci.vehicle.getFollower(rearVehicle[0])
            if rearVehicle2:
                if set(rearVehicle2) & set(["", None]):
                    rearVehicle2=("no_vehicle",0)
                else:
                    traci.vehicle.setColor(rearVehicle2[0], color=(0, 165,255 , 255))

        followerleft , leaderleft = self.getLeadFollowerLane2(vehicle_ids)
        intersectionVehicle = None
        for vehicle in vehicle_ids:
            if traci.vehicle.getRoadID(vehicle) == ":n2_0":
                intersectionVehicle= vehicle
        if intersectionVehicle and self.rl_car_distance <=150 and self.rl_car_distance >50 :
            if intersectionVehicle not in ["",None, self.rl_car_id]:
                vehicleDistance = traci.vehicle.getDrivingDistance(intersectionVehicle, "e35", 0.0)
                distanceVehicleRl = vehicleDistance - self.rl_car_distance
                leaderleft = (intersectionVehicle, distanceVehicleRl)
                print(leaderleft,'intersectionVehicle')
        
        ObservedVehicleLane0.append(leadVehicle)
        ObservedVehicleLane0.append(leadVehicle2)
        ObservedVehicleLane0.append(rearVehicle)
        ObservedVehicleLane0.append(rearVehicle2)
        ObservedVehicleLane1.append(followerleft)
        ObservedVehicleLane1.append(leaderleft)

        #elif traci.vehicle.getRoadID(self.rl_car_id) == "e23":
        return ObservedVehicleLane0, ObservedVehicleLane1
    def getLeadFollowerLane2(self, vehicle_ids):

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
                if vehicleLane.split('_')[1] == "1": # lane1
                    if distaneVehicleRl > 0:  # rear car
                        rearVehicles.append((vehicle, distaneVehicleRl))

                    elif distaneVehicleRl < 0:  ## lead car
                        leadVehicles.append((vehicle, distaneVehicleRl))

        if leadVehicles:
            leadVehicle = sorted(leadVehicles, key=lambda leadVehicles: leadVehicles[1])[-1]
            traci.vehicle.setColor(leadVehicle[0], color=(255, 0, 255, 255))
        else:
            leadVehicle = ('no_vehicle', 400)
            #print(leadVehicle)
        if rearVehicles:
            rearVehicle = sorted(rearVehicles, key=lambda rearVehicles: rearVehicles[1])[0]
            traci.vehicle.setColor(rearVehicle[0], color=(255, 255, 0, 255))


        else:
            rearVehicle = ('no_vehicle', 0)     
            #print(rearVehicle)
        return leadVehicle, rearVehicle

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
            traci.vehicle.setColor(leadVehicle[0], color=(0, 0, 255, 255))
        else:
            leadVehicle = ('no_vehicle', 400)
        if rearVehicles:
            rearVehicle = sorted(rearVehicles, key=lambda rearVehicles: rearVehicles[1])[0]
            traci.vehicle.setColor(rearVehicle[0], color=(0, 255, 0, 255))

        else:
            rearVehicle = ('no_vehicle', 0)
        return leadVehicle, rearVehicle

    def getObservations(self):
        # returns observations of the state

        state_speed = np.ones(7) * self.maxSpeed
        state_position_x = np.ones(7)
        state_position_y = np.ones(7)
        state_image = np.array(self.render())

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
            state_speed = np.clip(state_speed, 0, self.maxSpeed)
            state_position_x = np.clip(state_position_x, -abs(self.observation_space['xPos'].low),
                                       abs(self.observation_space['xPos'].high))
            state_position_y = np.clip(state_position_y, -abs(self.observation_space['yPos'].low),
                                       abs(self.observation_space['yPos'].high))
        state = {'image': state_image.astype(np.uint8),
                 'xPos': np.array(state_position_x, dtype=np.float32),
                 'yPos': np.array(state_position_y, dtype=np.float32),
                 'velocity': np.array(state_speed, dtype=np.float32)}
        return state

    def isTerminal(self):
        done = False
        whichterminal = 0
        if self.isCollision():
            done = True
            whichterminal = -1 # for collision
        elif traci.vehicle.getRoadID(self.rl_car_id) == 'e35':
            done = True
            whichterminal = +1 # for success
        elif traci.simulation.getTime() ==600:
            done = True
        if done == True:
            self.rl_car_id = None
        return done, whichterminal

    def apply_rl_action(self, action):


        rl_car_acc_duration = traci.simulation.getDeltaT()
        
        #print(rl_car_acc_duration,'rl simulation duration')
        rl_car_acc = action
        self.rl_car_id = self.addRLVehicle()
        currentSpeed = traci.vehicle.getSpeed(self.rl_car_id)
        next_vel = max([currentSpeed + rl_car_acc * rl_car_acc_duration, 0])
        self.stepRltime += 1
        # print(f"action {action}")
        # print(traci.vehicle.getPosition(self.rl_car_id))
        traci.vehicle.slowDown(self.rl_car_id, next_vel, rl_car_acc_duration)  # adjust acceleration of ego vehicle
        #traci.vehicle.setAcceleration(self.rl_car_id, action, rl_car_acc_duration)


    def step(self, action):

        rl_car_acc_duration = traci.simulation.getDeltaT()
        rl_car_acc = action
        # TODO add test baseline
        if not self.isBaseline:
            self.apply_rl_action(action)
        elif self.isBaseline:
            self.rl_car_id = self.addBaselineVehicle()
        # print(traci.vehicle.getRoadID(self.rl_car_id))
        self.simstep()
        if traci.vehicle.getRoadID(self.rl_car_id) == "e35":
            self.done = True
            if self.checkfirste35:
                self.info["mergeTime"] = traci.simulation.getTime()

                self.checkfirste35 = False

        else:
            self.time2merge += traci.simulation.getDeltaT()

        # traci.vehicle.setAcceleration(self.rl_car_id, rl_car_acc, rl_car_acc_duration)
        self.episodeLength -= 1

        reward = self.calc_reward()

        state = self.getObservations()
        if self.isCollision():
            rl_collide = 1
            self.done = True
            # print('collision - true')
        else:
            rl_collide = 0

        self.info["collision"] = rl_collide

        self.info["rlvehiclesSpawned"] = self.rl_vehiclesSpawned
        self.info["frame"] = state['image']
        self.done, _  = self.isTerminal()
        # if self.episodeLength == 0:
        #     self.done = True
            # self.close()

        return state, reward, self.done, self.info

    def close(self):
        self.closeSim()

    def calc_reward(self):

        vehicleAcc0 = []
        vehicleAcc1 = []
        rCongestion = 0
        rSuccess = 0

        reward = 0
        rTime = - abs(self.stepRltime) * self.rTimeAlpha

        rlPosition = traci.vehicle.getDistance(self.rl_car_id)
        # print(rlPosition, 'rlposition', self.alphaP, self.pastRlPosition)
        # rP = self.alphaP * np.sqrt(rlPosition - self.pastRlPosition)
        # self.pastRlPosition = rlPosition
        # rP = np.clip(rP, 1e-4,10)
        # rP = np.nan_to_num(rP)
        rP = 0

        # if self.isCollision():
        #    rCollision = self.rC
        # else:
        #    rCollision = 0

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

            reward = (rj + rd * self.alphaD + rTime + rP) / self.rSuccess
            # print(reward, rj, rd, 'reward before merging')
            # print(reward)
            return reward
        else:  # inmerge or post merge
            rearvehicle = traci.vehicle.getFollower(self.rl_car_id)
            leadvehicle = traci.vehicle.getLeader(self.rl_car_id)
            if rearvehicle and leadvehicle:
                rd = 2 - (abs(leadvehicle[1]) / self.maxDistance) ** self.alphaDistancef - abs(
                    rearvehicle[1] / self.maxDistance) ** self.alphaDistancer
            else:
                rd = 0
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
            else:
                rO = 0
            if rlLane == "e35" and self.checkSuccessOnce == 0:
                self.checkSuccessOnce = 1
            if self.checkSuccessOnce == 1:
                rSuccess = self.rSuccess
                self.checkSuccessOnce += 1

            else:
                rSuccess = 0

            if self.isCollision():
                rCollision = self.rC
                rSuccess = 0
            else:
                rCollision = 0
            reward = (rj + rCongestion + rSuccess + rCollision + rO + self.alphaD * np.real(
                rd) + rTime + rP) / self.rSuccess
            # print(reward, 'reward', rj,'rj', rCongestion, 'rCongestion', rSuccess, 'rsuc', rCollision, 'rCollision',
            #       rO, 'rO', rd, 'rd', rTime, 'rtime')

            # print(reward, 'reward final')
            # print(reward)

            return reward

    #
    def getStepCount(self):
        getTime = traci.simulation.getTime()
        return int(getTime)

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
            self.rl_car_id = "rl_car_" + str(1) # str(np.random.randint(1e12))  # random generate a rl_car
            errorAdd = traci.vehicle.add(self.rl_car_id, routeID="route_ramp",departSpeed = "random")  # vehicle add
            
            #print(errorAdd, '\n vehicle added \n\n\n' )
            traci.vehicle.setColor(self.rl_car_id, color=(255, 0, 0, 255))  # change vehicle color to red
            # print('type_id ', traci.vehicle.getTypeID(self.rl_car_id))
            traci.vehicle.setSpeedMode(self.rl_car_id, 32)  # change speed mode to 55 which is
            traci.vehicle.moveTo(self.rl_car_id, laneID = 'e42_0',#traci.vehicle.getLaneID(self.rl_car_id),
                    pos = 10)
            
            #print(self.max_acc, self.min_acc)
            traci.vehicle.setAccel(self.rl_car_id, self.max_acc)
            traci.vehicle.setDecel(self.rl_car_id,abs( self.min_acc))
            self.previoussteeringAngle = 0
            self.checkSuccessOnce = 0
            self.stepRltime = 0
            self.rl_vehiclesSpawned += 1
            self.time2merge = 0
            self.checkfirste35 = True
            self.pastRlPosition = 0
        return self.rl_car_id

    def addBaselineVehicle(self):
        'checks id no rl car is present in vehicle then addsit'
        vehicle_ids = self.getVehicleIds()
        if self.rl_car_id not in vehicle_ids:
            self.rl_car_id = "rl_car_" + str(np.random.randint(1e10))  # random generate a rl_car

            traci.vehicle.add(self.rl_car_id, routeID="route_ramp", typeID="KraussPassenger",
                    departSpeed="random")  # vehicle add
            traci.vehicle.setColor(self.rl_car_id, color=(255, 0, 0, 255))  # change vehicle color to blue
            # # print('type_id ', traci.vehicle.getTypeID(self.rl_car_id))
            traci.vehicle.setSpeedMode(self.rl_car_id, 31)  # change speed mode to 55 which is
            traci.vehicle.moveTo(self.rl_car_id, laneID = 'e42_0',#traci.vehicle.getLaneID(self.rl_car_id),
                    pos = 10)

            self.previoussteeringAngle = 0

            self.checkSuccessOnce = 0
            self.stepRltime = 0
            self.rl_vehiclesSpawned += 1
            self.time2merge = 0
            self.checkfirste35 = True
            self.pastRlPosition = 0

        if len(vehicle_ids) != 0:
            vehType = traci.vehicle.getTypeID(vehicle_ids[0])

        return self.rl_car_id

    def closeSim(self):
        traci.close()
        try:
            self.disp.stop()
        except AttributeError:
            pass


#
# # simpath = "/home/gauravb/Documents/MichiganTech/Programming/CustomRampTraining/custom_envs/sumo_Config/ramp_1.sumocfg"
# env = SumoRampEnv()


# register env


# Vecortized Env Create
from pettingzoo import AECEnv



