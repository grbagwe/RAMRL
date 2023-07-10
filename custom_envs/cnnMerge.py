import os
from abc import ABC

import gym
from custom_envs.gymsumo import SumoRamp
import traci
import numpy as np
from custom_envs.bsmMerge import BsmMerge, BsmMergeAllRewards



class CNNMerge(BsmMerge):
    def __init__(self, weights=None, action_space={"high": 3, "low": -4.5},
                 sumoParameters=None, isBaseline=False, obsspaces=None, render=0):
        self.min_acc = -1 * abs(action_space['low'])
        self.max_acc = action_space['high']
        self.oldAcc = self.min_acc

        self.obsspaces = obsspaces

        self.observation_space = obsspaces
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
        state_image = np.array(self.render())
        # if vehicle_ids:
        #     obsLane0, obsLane1 = self.getobservedVehicles(vehicle_ids)
        #
        #     for i, vehicle in enumerate(obsLane0):
        #         maxSpeed = traci.vehicle.getMaxSpeed(vehicle_ids[0])
        #         state_speed[i] = traci.vehicle.getSpeed(vehicle[0])
        #         state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
        #         state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
        #     for i, vehicle in enumerate(obsLane1, len(obsLane0)):
        #         state_speed[i] = traci.vehicle.getSpeed(vehicle[0])
        #
        #         state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
        #         state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
        #
        #     # rl state information
        #     state_speed[-1] = traci.vehicle.getSpeed(self.rl_car_id) / self.maxSpeed
        #     state_position_x[-1] = traci.vehicle.getPosition(self.rl_car_id)[0]
        #     state_position_y[-1] = traci.vehicle.getPosition(self.rl_car_id)[1]
        #     state_speed = np.clip(state_speed, 0, self.maxSpeed)
        #     state_position_x = np.clip(state_position_x, -abs(self.observation_space['xPos'].low),
        #                                abs(self.observation_space['xPos'].high))
        #     state_position_y = np.clip(state_position_y, -abs(self.observation_space['yPos'].low),
        #                                abs(self.observation_space['yPos'].high))
        state = state_image.astype(np.uint8)  # {
        # 'image': ,
        # 'xPos': np.array(state_position_x, dtype=np.float32),
        #  'yPos': np.array(state_position_y, dtype=np.float32),
        #  'velocity': np.array(state_speed, dtype=np.float32)}
        return state


class CNNMergeAllRewards:
    pass


class CNNMergeAllRewards(BsmMergeAllRewards):
    def __init__(self, weights=None, action_space={"high": 3, "low": -4.5},
                 sumoParameters=None, isBaseline=False, obsspaces=None, render=0):
        self.min_acc = -1 * abs(action_space['low'])
        self.max_acc = action_space['high']
        self.oldAcc = self.min_acc

        self.obsspaces = obsspaces

        self.observation_space = obsspaces
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
        state_image = np.array(self.render())
        # if vehicle_ids:
        #     obsLane0, obsLane1 = self.getobservedVehicles(vehicle_ids)
        #
        #     for i, vehicle in enumerate(obsLane0):
        #         maxSpeed = traci.vehicle.getMaxSpeed(vehicle_ids[0])
        #         state_speed[i] = traci.vehicle.getSpeed(vehicle[0])
        #         state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
        #         state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
        #     for i, vehicle in enumerate(obsLane1, len(obsLane0)):
        #         state_speed[i] = traci.vehicle.getSpeed(vehicle[0])
        #
        #         state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
        #         state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
        #
        #     # rl state information
        #     state_speed[-1] = traci.vehicle.getSpeed(self.rl_car_id) / self.maxSpeed
        #     state_position_x[-1] = traci.vehicle.getPosition(self.rl_car_id)[0]
        #     state_position_y[-1] = traci.vehicle.getPosition(self.rl_car_id)[1]
        #     state_speed = np.clip(state_speed, 0, self.maxSpeed)
        #     state_position_x = np.clip(state_position_x, -abs(self.observation_space['xPos'].low),
        #                                abs(self.observation_space['xPos'].high))
        #     state_position_y = np.clip(state_position_y, -abs(self.observation_space['yPos'].low),
        #                                abs(self.observation_space['yPos'].high))
        state = state_image.astype(np.uint8)  # {
        # 'image': ,
        # 'xPos': np.array(state_position_x, dtype=np.float32),
        #  'yPos': np.array(state_position_y, dtype=np.float32),
        #  'velocity': np.array(state_speed, dtype=np.float32)}
        return state
