import os
from abc import ABC

import gym
from custom_envs.gymsumo import SumoRamp
import traci
import numpy as np


class BsmMerge(SumoRamp):
    def __init__(self, weights=None, action_space={"high": 3, "low": -4.5},
                 sumoParameters=None, isBaseline=False, obsspaces=None, render=0):

        self.min_acc = -1 * abs(action_space['low'])
        self.max_acc = action_space['high']
        self.oldAcc = self.min_acc

        self.obsspaces = obsspaces

        self.observation_space = obsspaces #gym.spaces.Dict(obsspaces)
        self.action_space = gym.spaces.Box(low=np.array([self.min_acc]), high=np.array([self.max_acc]),
                                           dtype=np.float32)
        self.rewards = None
        self.pastRlspeed = 0

        self.label = str(SumoRamp.CONNECTION_LABEL)
        SumoRamp.CONNECTION_LABEL += 1
        print('weights', weights)
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
        self.SUMO_HOME = self.isSUMOHOME()
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

        self.sumoConfigFile = "./custom_envs/sumo_ConfigTaper/ramp_2.sumocfg"

        self.sumoBinary = os.path.join(self.SUMO_HOME, "bin/sumo-gui")

    def getObservations(self):
        # returns observations of the state

        state_speed = np.ones(7) * self.maxSpeed
        state_position_x = np.ones(7)
        state_position_y = np.ones(7)

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
#           print(self.observation_space.low[0])

            state_position_x = np.clip(state_position_x, -abs(self.observation_space.low[0]),
                                       abs(self.observation_space.high[0]))
            state_position_y = np.clip(state_position_y, -abs(self.observation_space.low[0]),
                                       abs(self.observation_space.high[0]))
        #state = {'xPos': np.array(state_position_x, dtype=np.float32),
        #         'yPos': np.array(state_position_y, dtype=np.float32),
        #         'velocity': np.array(state_speed, dtype=np.float32)}
#       print(state_position_x, state_position_y, state_speed)

        state = np.concatenate((state_position_x,state_position_y, state_speed))
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
        self.episodeLength -= 1

        state = self.getObservations()
        reward = self.getrewards()
        done, whichterminal = self.isTerminal()
        self.simstep()
        self.simstep()
        info = {
            'terminal': int(whichterminal),
            'mergeTime': self.episodeLength
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

    def getAccelerationReward(self):
        reward = 0
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
        return rj

    def getBeforeMergeReward(self):  # rewards before merge
        reward = 0
        leadVehicle, rearVehicle = self.getLeadFollowercarBMerge(self.getVehicleIds())
        leaddistance = leadVehicle[1]
        rearDistance = rearVehicle[1]
        rd = 2 - (abs(leaddistance) / self.maxDistance) ** self.alphaDistancef - (
                rearDistance / self.maxDistance) ** self.alphaDistancer
        rj = self.getAccelerationReward()

        reward = (rj + rd * self.alphaD)

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
        reward = lead_acc_1 + lead_acc_2 + follower_acc_1 + follower_acc_2
        # print(f" \n reward after merge {reward}")
        return reward

    def merge_timeReward(self):
        self.stepRltime += 0.1
        reward = self.stepRltime * self.rTimeAlpha
        return reward

    def getrewards(self):

        reward = 0
        rlLane = traci.vehicle.getRoadID(self.rl_car_id)


        if self.isCollision():
            reward = -abs(self.rC)  # collision penality
            print(f"reward collision {reward}")
        elif rlLane == 'e35':
            reward = +abs(self.rSuccess)
            print(f"reward success {reward}")
        else:  # is not terminal get the intermidiate reward for partial success
            # reward = self.getSpeedReward()  # +1 for positive speed and -1 for decreasing speed
            if rlLane == 'e42':  # before merge
                reward = self.getBeforeMergeReward()
                # print('reward before merge', reward)
            else:  # in the merge lane
                reward = self.getAfterMergeReward()
        reward = reward - abs(self.merge_timeReward()) # merge time penality
        print('intermidiate final reward', reward)

        return reward
