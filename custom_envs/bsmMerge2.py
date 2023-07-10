import os, sys
from abc import ABC

import gym
from custom_envs.gymsumo import SumoRamp
import traci
import numpy as np
from PIL import Image
from typing import Callable, Optional, Tuple, Union


if 'SUMO_HOME' in os.environ:
    SUMO_HOME = os.environ['SUMO_HOME']
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    print('sumo_loaded')
tools = os.path.join(SUMO_HOME, 'tools')
sys.path.append(tools)
#print(tools)

import traci as traci
import sumolib
LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ




class BsmMerge(SumoRamp):
    metadata = {'render.modes': ['human']}
    CONNECTION_LABEL = 0

    def __init__(self, weights=None, action_space={"high": 3, "low": -4.5},
                 sumoParameters=None, isBaseline=False, obsspaces=None, render=0, sumo_seed: Union[str, int] = 'random',
                 sumo_warnings: bool = True, net_file: str = './custom_envs/sumo_ConfigTaper/Ramp_2.net.xml',
                 route_file: str = './custom_envs/sumo_ConfigTaper/ramp_2.rou.xml'):
        self.min_acc = -1 * abs(action_space['low'])
        self.max_acc = action_space['high']
        self.oldAcc = self.min_acc

        self.obsspaces = obsspaces

        self.observation_space = gym.spaces.Dict(obsspaces)
        self.action_space = gym.spaces.Box(low=np.array([self.min_acc]), high=np.array([self.max_acc]),
                                           dtype=np.float32)
        self.rewards = None
        self.pastRlspeed = 0

        self.sumo_seed = sumo_seed
        self.sumo_warnings = sumo_warnings
        self.use_gui = True
        self._net = net_file
        self._route = route_file

        if self.use_gui:
            self._sumo_binary= sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary= sumolib.checkBinary('sumo')


        self.label = str(SumoRamp.CONNECTION_LABEL)
        SumoRamp.CONNECTION_LABEL += 1
        self.sumo = None
        # self.SUMO_HOME = self.isSUMOHOME()
        if LIBSUMO:
            traci.start(
                [sumolib.checkBinary('sumo'), '-n', self._net])  # Start only to retrieve traffic light information
            conn = traci
        else:
            traci.start([sumolib.checkBinary('sumo'), '-n', self._net], label='init_connection' + self.label)
            conn = traci.getConnection('init_connection' + self.label)
        conn.close()


        self.alphasl0 = weights['alphasl0']  # 0.7
        self.alphasl1 = weights['alphasl1']  # 0.2
        self.rSuccess = weights['rSuccess']  # 50
        self.alphaO = weights['alphaO']  # 0.03
        self.rTimeAlpha = weights['rTimeAlpha']  # 0.001
        self.alphaD = weights['alphaD']  # 0.001

        self.maxEpisodeLength = sumoParameters['episodeLength']  # 3000  # 3600

        self.virtual_display = (1024, 1024)
        self.image_shape = (200, 768)  # 512,768
        # self.image_resize = (512, 512)

        self.episodeLength = self.maxEpisodeLength
        self.rC = weights['rC']  # -150
        self.maxSpeed = 30  # traci.vehicle.getMaxSpeed#30
        self.alphaJ = weights['alphaJ']  # 0.5
        self.run = 0
        self.maxDistance = 400  # weights['maxDistance']  # 400
        self.alphaDistancef = self.alphaDistancer = weights['alphaDistance']  # .02
        self.pastRlPosition = 0
        self.alphaP = weights['alphaP']  # 0.01
        self.stepRltime  = 0

        # min and max acceleration
        self.observationDistance = 100  # observation distance for the rl vehicle

        self.min_acc = -1 * abs(action_space['low'])
        self.max_acc = action_space['high']
        self.oldAcc = self.min_acc
        self.isBaseline = isBaseline

        self.env_render = render

        # self.sumoConfigFile = "./custom_envs/sumo_ConfigTaper/ramp_2.sumocfg"
        #
        # self.sumoBinary = os.path.join(self.SUMO_HOME, "bin/sumo-gui")

    def getObservations(self):
        # returns observations of the state

        state_speed = np.ones(7) * self.maxSpeed
        state_position_x = np.ones(7)
        state_position_y = np.ones(7)

        vehicle_ids = self.getVehicleIds()
        if vehicle_ids:
            #for vehicle in vehicle_ids:
            #    traci.vehicle.setColor(vehicle, color=(255, 255, 255, 255))  # change vehicle color to blue

            obsLane0, obsLane1 = self.getobservedVehicles(vehicle_ids)

            for i, vehicle in enumerate(obsLane0):
                
                maxSpeed = traci.vehicle.getMaxSpeed(vehicle_ids[0])
                state_speed[i] = traci.vehicle.getSpeed(vehicle[0])
                state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
                traci.vehicle.setColor(vehicle[0], color=(255, 0, 255, 255))  # change vehicle color to blue

            for i, vehicle in enumerate(obsLane1, len(obsLane0)):
                state_speed[i] = traci.vehicle.getSpeed(vehicle[0])

                state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
                traci.vehicle.setColor(vehicle[0], color=(255, 255, 0, 255))  # change vehicle color to blue
                #print(f"\n\n\n vehicle id {vehicle[0]}")

            # rl state information
            state_speed[-1] = traci.vehicle.getSpeed(self.rl_car_id) / self.maxSpeed
            state_position_x[-1] = traci.vehicle.getPosition(self.rl_car_id)[0]
            state_position_y[-1] = traci.vehicle.getPosition(self.rl_car_id)[1]
            state_speed = np.clip(state_speed, 0, self.maxSpeed)
            state_position_x = np.clip(state_position_x, -abs(self.observation_space['xPos'].low),
                                       abs(self.observation_space['xPos'].high))
            state_position_y = np.clip(state_position_y, -abs(self.observation_space['yPos'].low),
                                       abs(self.observation_space['yPos'].high))
        state = {'xPos': np.array(state_position_x, dtype=np.float32),
                 'yPos': np.array(state_position_y, dtype=np.float32),
                 'velocity': np.array(state_speed, dtype=np.float32)}
        return state

    def getrewards(self):

        if traci.vehicle.getSpeed(self.rl_car_id) - self.pastRlspeed > 0:
            reward = 1
        else:
            reward = -1
        self.pastRlspeed = traci.vehicle.getSpeed(self.rl_car_id)

        return reward

    def step(self, action):
        '''
        _______
        Parameters
        action : action given to the rl agent

        Returns
        -------
        observation : array_like
            observation of the current environment
        reward : float
            reward associated with the previous state/action pair
        done : bool
            indicates whether the episode has ended
        info : dict
            contains other diagnostic information from the previous action
        '''
        if not self.isBaseline:
            self.apply_rl_action(action)
        else:
            self.addBaselineVehicle()
        

        self.simstep()
        
        state = self.getObservations()
        reward = self.getrewards()
        done, whichterminal = self.isTerminal()
        info = {
            'terminal': int(whichterminal),
            'mergeTime': traci.simulation.getTime()
        }

        return state, reward, done, info


class BsmMergeAllRewards(BsmMerge):

    def getSpeedReward(self):
        if traci.vehicle.getSpeed(self.rl_car_id) - self.pastRlspeed > 0:
            reward = 1
        else:
            reward = -1
        self.pastRlspeed = traci.vehicle.getSpeed(self.rl_car_id)

        return reward

    def getAccelerationReward(self): # acceleartion penality and reward
        reward = traci.vehicle.getAcceleration(self.rl_car_id)/ 3#traci.vehicle.getAccel(self.rl_car_id)
        return reward

    def getTimeReward(self):  # penalty for time spend on the road
        reward = 0
        return reward

    def getAccJerkReward(self):
        rlAcceleration = traci.vehicle.getAcceleration(self.rl_car_id)
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

        #print('rj', rj)

        return rj

    def getBeforeMergeReward(self):  # rewards before merge
        reward = 0
        leadVehicle, rearVehicle = self.getLeadFollowercarBMerge(self.getVehicleIds())
        leaddistance = leadVehicle[1]
        rearDistance = rearVehicle[1]
        rd = 2 - (abs(leaddistance) / self.maxDistance) ** self.alphaDistancef - (
                rearDistance / self.maxDistance) ** self.alphaDistancer
        rj = self.getAccJerkReward()
        rAccel = self.getAccelerationReward()

        reward = rj + rd* self.alphaD + rAccel
        #print('rd', rd , 'rd *  alpa ' , rd*self.alphaD)
        #print( rj , 'rj',rAccel,'r Accel')
        #print(reward)

        return reward

    def getAfterMergeReward(self):  # reward after the vehicle has merged
        rj = self.getAccelerationReward()  # we still get the jerk reward
        lead_acc_1 = lead_acc_2 = follower_acc_1 = follower_acc_2 = 0

        leadvehicle = traci.vehicle.getLeader(self.rl_car_id)

        if leadvehicle != None:
            leadvehicle2 = traci.vehicle.getLeader(leadvehicle[0])
            lead_acc_1 = traci.vehicle.getAcceleration(leadvehicle[0]) / traci.vehicle.getAccel(self.rl_car_id)
            if leadvehicle2 != None:
                lead_acc_2 = traci.vehicle.getAcceleration(leadvehicle2[0]) / traci.vehicle.getAccel(self.rl_car_id)
        followervehicle = traci.vehicle.getLeader(self.rl_car_id)
        if  followervehicle != None:
            followervehicle2 = traci.vehicle.getLeader(followervehicle[0])
            follower_acc_1 = traci.vehicle.getAcceleration(followervehicle[0]) / traci.vehicle.getAccel(self.rl_car_id)
            if  followervehicle2 != None:
                follower_acc_2 = traci.vehicle.getAcceleration(followervehicle2[0]) / traci.vehicle.getAccel(self.rl_car_id)

        rAccel = self.getAccelerationReward()
        reward = lead_acc_1 + lead_acc_2 + follower_acc_1 + follower_acc_2 + rAccel
        # print(f" \n reward after merge {reward}")
        #print('reward after merge :', reward)
        #print('after merge reward',reward, 'rAccel' , round(rAccel,3), 'vehicleAccel', lead_acc_1 , lead_acc_2 , follower_acc_1 , follower_acc_2)
        return reward

    def merge_timeReward(self):
        self.stepRltime = 0.1
        reward = self.stepRltime * self.rTimeAlpha
        #print('merge_ time reward',reward)

        return reward

    def getrewards(self):

        reward = 0
        rlLane = traci.vehicle.getRoadID(self.rl_car_id)


        if self.isCollision():
            reward = -abs(self.rC)  # collision penality
            #print(f"reward collision {reward}")
        elif rlLane == 'e35':
            reward = +abs(self.rSuccess)
            #print(f"reward success {reward}")
        elif self.episodeLength ==0:
            reward = -1 * abs(self.rSuccess)/2
        else:  # is not terminal get the intermidiate reward for partial success
            # reward = self.getSpeedReward()  # +1 for positive speed and -1 for decreasing speed
            if rlLane == 'e42':  # before merge
                reward = self.getBeforeMergeReward()
                #print('reward before merge', reward)
            else:  # in the merge lane
                reward = self.getAfterMergeReward()
        reward = reward - abs(self.merge_timeReward()) # merge time penality
        #print('intermidiate final reward', reward)

        return reward
