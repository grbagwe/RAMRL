import os
from abc import ABC

import gym
from custom_envs.gymsumo import SumoRamp
import traci
import numpy as np
from custom_envs.bsmMerge import BsmMerge, BsmMergeAllRewards
from typing import Callable, Optional, Tuple, Union

class MultiMerge(BsmMerge):
    def getObservations(self):
        # returns observations of the state

        state_speed = np.ones(7) * self.maxSpeed
        state_position_x = np.ones(7)
        state_position_y = np.ones(7)

        vehicle_ids = self.getVehicleIds()
        state_image = np.array(self.render())
        if vehicle_ids:
            obsLane0, obsLane1 = self.getobservedVehicles(vehicle_ids)

            for i, vehicle in enumerate(obsLane0):
                maxSpeed = traci.vehicle.getMaxSpeed(vehicle_ids[0])
                if vehicle:

                    if vehicle[0] not in ["no_vehicle","", None]:

                        state_speed[i] = traci.vehicle.getSpeed(vehicle[0])
                        state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                        state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
            for i, vehicle in enumerate(obsLane1, len(obsLane0)):
                if vehicle:
                    if vehicle not in ["no_vehicle","", None]:
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
        state = {
                'image': state_image.astype(np.uint8),
                'xPos': np.array(state_position_x, dtype=np.float32),
                 'yPos': np.array(state_position_y, dtype=np.float32),
                 'velocity': np.array(state_speed, dtype=np.float32)}

        return state

class MultiMergeAllRewards(BsmMergeAllRewards):
    def getObservations(self):
        # returns observations of the state

        state_speed = np.ones(7) * self.maxSpeed
        state_position_x = np.ones(7)
        state_position_y = np.ones(7)

        vehicle_ids = self.getVehicleIds()
        state_image = np.array(self.render())
        if vehicle_ids:
            for vehicle in vehicle_ids:
                if not "rl" in vehicle:
                    traci.vehicle.setColor(vehicle, color=(255, 255, 255, 255))  # change vehicle color to white

            obsLane0, obsLane1 = self.getobservedVehicles(vehicle_ids)


            for i, vehicle in enumerate(obsLane0):
                maxSpeed = traci.vehicle.getMaxSpeed(vehicle_ids[0])
                #print(vehicle)
                if vehicle:
                    if vehicle[0] not in ["no_vehicle","", None]:
                        state_speed[i] = traci.vehicle.getSpeed(vehicle[0])
                        state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                        state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
                #traci.vehicle.setColor(vehicle[0], color=(255, 0, 255, 255))  # change vehicle color to blue

            for i, vehicle in enumerate(obsLane1, len(obsLane0)):
                #print(vehicle)
                if vehicle:

                    if vehicle[0] not in ["no_vehicle","", None]:
                        state_speed[i] = traci.vehicle.getSpeed(vehicle[0])

                        state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                        state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
                #traci.vehicle.setColor(vehicle[0], color=(255, 255, 0, 255))  # change vehicle color to blue

            # rl state information
            state_speed[-1] = traci.vehicle.getSpeed(self.rl_car_id) / self.maxSpeed
            state_position_x[-1] = traci.vehicle.getPosition(self.rl_car_id)[0]
            state_position_y[-1] = traci.vehicle.getPosition(self.rl_car_id)[1]
            state_speed = np.clip(state_speed, 0, self.maxSpeed)
            state_position_x = np.clip(state_position_x, -abs(self.observation_space['xPos'].low),
                    abs(self.observation_space['xPos'].high))
            state_position_y = np.clip(state_position_y, -abs(self.observation_space['yPos'].low),
                                       abs(self.observation_space['yPos'].high))
        state = {
                'image': state_image.astype(np.uint8),
                'xPos': np.array(state_position_x, dtype=np.float32),
                 'yPos': np.array(state_position_y, dtype=np.float32),
                 'velocity': np.array(state_speed, dtype=np.float32)}
        return state
