import os
from abc import ABC

import gym
from custom_envs.gymsumo import SumoRamp
import traci
import numpy as np
from custom_envs.bsmMerge import BsmMerge, BsmMergeAllRewards
from typing import Callable, Optional, Tuple, Union
from scipy.ndimage.filters import gaussian_filter

noise_level = 0.25  # noise level percent


class MultiMerge(BsmMerge):
    def getObservations(self):
        # returns observations of the state

        state_speed = np.ones(7) * self.maxSpeed
        state_position_x = np.ones(7)
        state_position_y = np.ones(7)
        state_acc = np.zeros(7)


        vehicle_ids = self.getVehicleIds()
        state_image = np.array(self.render())
        if vehicle_ids:
            obsLane0, obsLane1 = self.getobservedVehicles(vehicle_ids)

            for i, vehicle in enumerate(obsLane0):
                maxSpeed = traci.vehicle.getMaxSpeed(vehicle_ids[0])
                if vehicle:

                    if vehicle[0] not in ["no_vehicle","", None]:

                        state_speed[i] = traci.vehicle.getSpeed(vehicle[0])
                        state_acc[i]   = traci.vehicle.getAcceleration(vehicle[0])
                        state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                        
                        state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
            for i, vehicle in enumerate(obsLane1, len(obsLane0)):
                if vehicle:
                    if vehicle not in ["no_vehicle","", None]:
                        state_speed[i] = traci.vehicle.getSpeed(vehicle[0])
                        state_acc[i]   = traci.vehicle.getAcceleration(vehicle[0])
                        state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                        state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]

            # rl state information
            state_speed[-1] = traci.vehicle.getSpeed(self.rl_car_id)
            state_position_x[-1] = traci.vehicle.getPosition(self.rl_car_id)[0]
            state_position_y[-1] = traci.vehicle.getPosition(self.rl_car_id)[1]
            state_acc[-1]        = traci.vehicle.getAcceleration(self.rl_car_id)

            #state_speed = np.clip(state_speed, 0, self.maxSpeed)
            #state_position_x = np.clip(state_position_x, -abs(self.observation_space['xPos'].low),
             #                          abs(self.observation_space['xPos'].high))
            #state_position_y = np.clip(state_position_y, -abs(self.observation_space['yPos'].low),
            #                           abs(self.observation_space['yPos'].high))
        state = {
                'image': state_image.astype(np.uint8),
                'xPos': np.array(state_position_x, dtype=np.float32),
                 'yPos': np.array(state_position_y, dtype=np.float32),
                 'velocity': np.array(state_speed, dtype=np.float32),
                 'acceleration' : np.array(state_acc, dtype= np.float32)
                 }

        return state

class MultiMergeAllRewards(BsmMergeAllRewards):
    def getObservations(self):
        # returns observations of the state

        state_speed = np.ones(7) * self.maxSpeed
        state_position_x = np.ones(7)
        state_position_y = np.ones(7)

        state_acc = np.zeros(7)
        state_latSpeed = np.zeros(7)
        vehicle_ids = self.getVehicleIds()
        state_image = np.array(self.render())
        if vehicle_ids:
            for vehicle in vehicle_ids:
                if not "rl" in vehicle:
                    traci.vehicle.setColor(vehicle, color=(255, 255, 255, 255))  # change vehicle color to white

            obsLane0, obsLane1 = self.getobservedVehicles(vehicle_ids)


            for i, vehicle in enumerate(obsLane0):
                maxSpeed = traci.vehicle.getMaxSpeed(vehicle_ids[0])
                if vehicle:
                    if vehicle[0] not in ["no_vehicle","", None]:
                        state_speed[i] = traci.vehicle.getSpeed(vehicle[0])
                        state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                        state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
                        state_acc[i]   = traci.vehicle.getAcceleration(vehicle[0])
                        state_latSpeed[i] = traci.vehicle.getLateralSpeed(vehicle[0])

                        
            for i, vehicle in enumerate(obsLane1, len(obsLane0)):
                #print(vehicle)
                if vehicle:

                    if vehicle[0] not in ["no_vehicle","", None]:
                        state_speed[i] = traci.vehicle.getSpeed(vehicle[0])

                        state_acc[i]   = traci.vehicle.getAcceleration(vehicle[0])
                        state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                        state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
            
            state_speed[-1] = traci.vehicle.getSpeed(self.rl_car_id) #/ self.maxSpeed
            state_position_x[-1] = traci.vehicle.getPosition(self.rl_car_id)[0]
            state_position_y[-1] = traci.vehicle.getPosition(self.rl_car_id)[1]
            state_acc[-1]        = traci.vehicle.getAcceleration(self.rl_car_id)
            state_latSpeed[-1] = traci.vehicle.getLateralSpeed(self.rl_car_id)


         

        state = {
                'image': state_image.astype(np.uint8),
                'xPos': np.array(state_position_x, dtype=np.float32),
                 'yPos': np.array(state_position_y, dtype=np.float32),
                 'velocity': np.array(state_speed, dtype=np.float32),
                 'acceleration' : np.array(state_acc, dtype= np.float32),
                 'latSpeed': np.array(state_latSpeed, dtype=np.float32),
                 }
        return state

class BSM_Noise_No_Image(BsmMergeAllRewards):
    def getObservations(self):
        # returns observations of the state

        state_speed = np.ones(7) * self.maxSpeed
        state_position_x = np.ones(7)
        state_position_y = np.ones(7)
        state_acc = np.zeros(7)
        state_latSpeed = np.zeros(7)
        vehicle_ids = self.getVehicleIds()
        if vehicle_ids:
            for vehicle in vehicle_ids:
                if not "rl" in vehicle:
                    traci.vehicle.setColor(vehicle, color=(255, 255, 255, 255))  # change vehicle color to white

            obsLane0, obsLane1 = self.getobservedVehicles(vehicle_ids)

            for i, vehicle in enumerate(obsLane0):
                maxSpeed = traci.vehicle.getMaxSpeed(vehicle_ids[0])
                if vehicle:
                    if vehicle[0] not in ["no_vehicle", "", None]:
                        state_speed[i] = traci.vehicle.getSpeed(vehicle[0])
                        state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                        state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
                        state_acc[i] = traci.vehicle.getAcceleration(vehicle[0])
                        state_latSpeed[i] = traci.vehicle.getLateralSpeed(vehicle[0])

            for i, vehicle in enumerate(obsLane1, len(obsLane0)):
                # print(vehicle)
                if vehicle:

                    if vehicle[0] not in ["no_vehicle", "", None]:
                        state_speed[i] = traci.vehicle.getSpeed(vehicle[0])

                        state_acc[i] = traci.vehicle.getAcceleration(vehicle[0])
                        state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                        state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
                        state_latSpeed[i] = traci.vehicle.getLateralSpeed(vehicle[0])


            state_speed[-1] = traci.vehicle.getSpeed(self.rl_car_id)  # / self.maxSpeed
            state_position_x[-1] = traci.vehicle.getPosition(self.rl_car_id)[0]
            state_position_y[-1] = traci.vehicle.getPosition(self.rl_car_id)[1]
            state_acc[-1] = traci.vehicle.getAcceleration(self.rl_car_id)

            state_latSpeed[-1] = traci.vehicle.getLateralSpeed(self.rl_car_id)
            
            state_speed = state_speed + np.random.normal(0, 0.277778, np.size(state_position_x)) * noise_level 
            state_position_x = state_position_x + np.random.normal(0, 1.5, np.size(state_position_x)) * noise_level 
            state_position_y = state_position_y + np.random.normal(0, 1.5, np.size(state_position_x)) * noise_level 

            state_acc = state_acc + np.random.normal(0, 0.3, np.size(state_position_x)) * noise_level 

            
            state_latSpeed =  state_latSpeed + np.random.normal(0, 0.277778, np.size(state_position_x)) * noise_level 

        state = {
            'xPos': np.array(state_position_x, dtype=np.float32),
            'yPos': np.array(state_position_y, dtype=np.float32),
            'velocity': np.array(state_speed, dtype=np.float32),
            'acceleration': np.array(state_acc, dtype=np.float32),
            'latSpeed': np.array(state_latSpeed, dtype=np.float32),
        }
        return state


class BSM_No_Image(BsmMergeAllRewards):
    def getObservations(self):
        # returns observations of the state

        state_speed = np.ones(7) * self.maxSpeed
        state_position_x = np.ones(7)
        state_position_y = np.ones(7)
        state_latSpeed = np.zeros(7)
        state_acc = np.zeros(7)
        vehicle_ids = self.getVehicleIds()
        if vehicle_ids:
            for vehicle in vehicle_ids:
                if not "rl" in vehicle:
                    traci.vehicle.setColor(vehicle, color=(255, 255, 255, 255))  # change vehicle color to white

            obsLane0, obsLane1 = self.getobservedVehicles(vehicle_ids)

            for i, vehicle in enumerate(obsLane0):
                maxSpeed = traci.vehicle.getMaxSpeed(vehicle_ids[0])
                if vehicle:
                    if vehicle[0] not in ["no_vehicle", "", None]:
                        state_speed[i] = traci.vehicle.getSpeed(vehicle[0])
                        state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                        state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
                        state_latSpeed[i] = traci.vehicle.getLateralSpeed(vehicle[0])
                        state_acc[i] = traci.vehicle.getAcceleration(vehicle[0])

            for i, vehicle in enumerate(obsLane1, len(obsLane0)):
                # print(vehicle)
                if vehicle:

                    if vehicle[0] not in ["no_vehicle", "", None]:
                        state_speed[i] = traci.vehicle.getSpeed(vehicle[0])

                        state_acc[i] = traci.vehicle.getAcceleration(vehicle[0])
                        state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                        state_latSpeed[i] = traci.vehicle.getLateralSpeed(vehicle[0])
                        state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
            state_speed[-1] = traci.vehicle.getSpeed(self.rl_car_id)  # / self.maxSpeed
            state_position_x[-1] = traci.vehicle.getPosition(self.rl_car_id)[0]
            state_position_y[-1] = traci.vehicle.getPosition(self.rl_car_id)[1]
            state_acc[-1] = traci.vehicle.getAcceleration(self.rl_car_id)

            state_latSpeed[-1] = traci.vehicle.getLateralSpeed(self.rl_car_id)
            #state_speed = state_speed + np.random.normal(0, 0.277778, np.size(state_position_x))
            #state_position_x = state_position_x + np.random.normal(0, 1.5, np.size(state_position_x))
            #state_position_y = state_position_y + np.random.normal(0, 1.5, np.size(state_position_x))
            #state_acc = state_acc + np.random.normal(0, 0.3, np.size(state_position_x))

        state = {
            'xPos': np.array(state_position_x, dtype=np.float32),
            'yPos': np.array(state_position_y, dtype=np.float32),
            'velocity': np.array(state_speed, dtype=np.float32),
            'acceleration': np.array(state_acc, dtype=np.float32),
            'latSpeed': np.array(state_latSpeed, dtype=np.float32),
        }
        return state

class Image_No_BSM(BsmMergeAllRewards):
    def getObservations(self):
        # returns observations of the state

        state_speed = np.ones(7) * self.maxSpeed
        state_position_x = np.ones(7)
        state_position_y = np.ones(7)

        state_acc = np.zeros(7)
        vehicle_ids = self.getVehicleIds()
        state_image = np.array(self.render())
        if vehicle_ids:
            for vehicle in vehicle_ids:
                if not "rl" in vehicle:
                    traci.vehicle.setColor(vehicle, color=(255, 255, 255, 255))  # change vehicle color to white

            obsLane0, obsLane1 = self.getobservedVehicles(vehicle_ids)

            for i, vehicle in enumerate(obsLane0):
                maxSpeed = traci.vehicle.getMaxSpeed(vehicle_ids[0])
                if vehicle:
                    if vehicle[0] not in ["no_vehicle", "", None]:
                        state_speed[i] = traci.vehicle.getSpeed(vehicle[0])
                        state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                        state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
                        state_acc[i] = traci.vehicle.getAcceleration(vehicle[0])

            for i, vehicle in enumerate(obsLane1, len(obsLane0)):
                # print(vehicle)
                if vehicle:

                    if vehicle[0] not in ["no_vehicle", "", None]:
                        state_speed[i] = traci.vehicle.getSpeed(vehicle[0])

                        state_acc[i] = traci.vehicle.getAcceleration(vehicle[0])
                        state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                        state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
            state_speed[-1] = traci.vehicle.getSpeed(self.rl_car_id)  # / self.maxSpeed
            state_position_x[-1] = traci.vehicle.getPosition(self.rl_car_id)[0]
            state_position_y[-1] = traci.vehicle.getPosition(self.rl_car_id)[1]
            state_acc[-1] = traci.vehicle.getAcceleration(self.rl_car_id)

        sigmavalue = 1


        state = {
            'image': state_image.astype(np.uint8),
        }
        return state

class Image_Noise_No_BSM(BsmMergeAllRewards):
    def getObservations(self):
        # returns observations of the state

        state_speed = np.ones(7) * self.maxSpeed
        state_position_x = np.ones(7)
        state_position_y = np.ones(7)

        state_acc = np.zeros(7)
        vehicle_ids = self.getVehicleIds()
        state_image = np.array(self.render())
        if vehicle_ids:
            for vehicle in vehicle_ids:
                if not "rl" in vehicle:
                    traci.vehicle.setColor(vehicle, color=(255, 255, 255, 255))  # change vehicle color to white

            obsLane0, obsLane1 = self.getobservedVehicles(vehicle_ids)

            for i, vehicle in enumerate(obsLane0):
                maxSpeed = traci.vehicle.getMaxSpeed(vehicle_ids[0])
                if vehicle:
                    if vehicle[0] not in ["no_vehicle", "", None]:
                        state_speed[i] = traci.vehicle.getSpeed(vehicle[0])
                        state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                        state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
                        state_acc[i] = traci.vehicle.getAcceleration(vehicle[0])

            for i, vehicle in enumerate(obsLane1, len(obsLane0)):
                # print(vehicle)
                if vehicle:

                    if vehicle[0] not in ["no_vehicle", "", None]:
                        state_speed[i] = traci.vehicle.getSpeed(vehicle[0])

                        state_acc[i] = traci.vehicle.getAcceleration(vehicle[0])
                        state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                        state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
            state_speed[-1] = traci.vehicle.getSpeed(self.rl_car_id)  # / self.maxSpeed
            state_position_x[-1] = traci.vehicle.getPosition(self.rl_car_id)[0]
            state_position_y[-1] = traci.vehicle.getPosition(self.rl_car_id)[1]
            state_acc[-1] = traci.vehicle.getAcceleration(self.rl_car_id)

        sigmavalue = 1 * noise_level 

        state_image = gaussian_filter(state_image, sigma=sigmavalue)


        state = {
            'image': state_image.astype(np.uint8),
        }
        return state

class BSM_Noise_Image_Noise(BsmMergeAllRewards):
    def getObservations(self):
        # returns observations of the state

        state_speed = np.ones(7) * self.maxSpeed
        state_position_x = np.ones(7)
        state_position_y = np.ones(7)

        state_latSpeed = np.zeros(7)
        state_acc = np.zeros(7)
        vehicle_ids = self.getVehicleIds()
        state_image = np.array(self.render())

        if vehicle_ids:
            for vehicle in vehicle_ids:
                if not "rl" in vehicle:
                    traci.vehicle.setColor(vehicle, color=(255, 255, 255, 255))  # change vehicle color to white

            obsLane0, obsLane1 = self.getobservedVehicles(vehicle_ids)

            for i, vehicle in enumerate(obsLane0):
                maxSpeed = traci.vehicle.getMaxSpeed(vehicle_ids[0])
                # print(vehicle)
                if vehicle:
                    if vehicle[0] not in ["no_vehicle", "", None]:
                        state_speed[i] = traci.vehicle.getSpeed(vehicle[0])
                        state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                        state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
                        state_latSpeed[i] = traci.vehicle.getLateralSpeed(vehicle[0])
                        state_acc[i] = traci.vehicle.getAcceleration(vehicle[0])
                # traci.vehicle.setColor(vehicle[0], color=(255, 0, 255, 255))  # change vehicle color to blue

            for i, vehicle in enumerate(obsLane1, len(obsLane0)):
                # print(vehicle)
                if vehicle:

                    if vehicle[0] not in ["no_vehicle", "", None]:
                        state_speed[i] = traci.vehicle.getSpeed(vehicle[0])

                        state_acc[i] = traci.vehicle.getAcceleration(vehicle[0])
                        state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                        state_latSpeed[i] = traci.vehicle.getLateralSpeed(vehicle[0])
                        state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
                # traci.vehicle.setColor(vehicle[0], color=(255, 255, 0, 255))  # change vehicle color to blue

            # rl state information
            state_speed[-1] = traci.vehicle.getSpeed(self.rl_car_id)  # / self.maxSpeed
            state_position_x[-1] = traci.vehicle.getPosition(self.rl_car_id)[0]
            state_position_y[-1] = traci.vehicle.getPosition(self.rl_car_id)[1]
            # state_speed = np.clip(state_speed, 0, self.maxSpeed)
            state_latSpeed[-1] = traci.vehicle.getLateralSpeed(self.rl_car_id)
            state_acc[-1] = traci.vehicle.getAcceleration(self.rl_car_id)
            # state_position_x = np.clip(state_position_x, -abs(self.observation_space['xPos'].low),
            #        abs(self.observation_space['xPos'].high))
            # state_position_y = np.clip(state_position_y, -abs(self.observation_space['yPos'].low),
            #                          abs(self.observation_space['yPos'].high))

        sigmavalue = 1 * noise_level 

        state_image = gaussian_filter(state_image, sigma=sigmavalue)

        state_speed = state_speed + np.random.normal(0, 0.277778,np.size(state_position_x)) * noise_level 

        state_position_x = state_position_x + np.random.normal(0, 1.5,np.size(state_position_x)) * noise_level 

        state_position_y = state_position_y + np.random.normal(0, 1.5,np.size(state_position_x)) * noise_level 

        state_acc = state_acc + np.random.normal(0, 0.3,np.size(state_position_x)) * noise_level 


        state_latSpeed =  state_latSpeed + np.random.normal(0, 0.277778, np.size(state_position_x))
        state = {
            'image': state_image.astype(np.uint8),
            'xPos': np.array(state_position_x, dtype=np.float32),
            'yPos': np.array(state_position_y, dtype=np.float32),
            'velocity': np.array(state_speed, dtype=np.float32),
            'latSpeed': np.array(state_latSpeed, dtype=np.float32),
            'acceleration': np.array(state_acc, dtype=np.float32),
        }
        return state

class BSM_Noise_Perfect_Image(BsmMergeAllRewards):
    def getObservations(self):
        # returns observations of the state

        state_speed = np.ones(7) * self.maxSpeed
        state_position_x = np.ones(7)
        state_position_y = np.ones(7)

        state_latSpeed = np.zeros(7)
        state_acc = np.zeros(7)
        vehicle_ids = self.getVehicleIds()
        state_image = np.array(self.render())

        if vehicle_ids:
            for vehicle in vehicle_ids:
                if not "rl" in vehicle:
                    traci.vehicle.setColor(vehicle, color=(255, 255, 255, 255))  # change vehicle color to white

            obsLane0, obsLane1 = self.getobservedVehicles(vehicle_ids)

            for i, vehicle in enumerate(obsLane0):
                maxSpeed = traci.vehicle.getMaxSpeed(vehicle_ids[0])
                # print(vehicle)
                if vehicle:
                    if vehicle[0] not in ["no_vehicle", "", None]:
                        state_speed[i] = traci.vehicle.getSpeed(vehicle[0])
                        state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                        state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
                        state_latSpeed[i] = traci.vehicle.getLateralSpeed(vehicle[0])
                        state_acc[i] = traci.vehicle.getAcceleration(vehicle[0])
                # traci.vehicle.setColor(vehicle[0], color=(255, 0, 255, 255))  # change vehicle color to blue

            for i, vehicle in enumerate(obsLane1, len(obsLane0)):
                # print(vehicle)
                if vehicle:

                    if vehicle[0] not in ["no_vehicle", "", None]:
                        state_speed[i] = traci.vehicle.getSpeed(vehicle[0])

                        state_acc[i] = traci.vehicle.getAcceleration(vehicle[0])
                        state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                        state_latSpeed[i] = traci.vehicle.getLateralSpeed(vehicle[0])
                        state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
                # traci.vehicle.setColor(vehicle[0], color=(255, 255, 0, 255))  # change vehicle color to blue

            # rl state information
            state_speed[-1] = traci.vehicle.getSpeed(self.rl_car_id)  # / self.maxSpeed
            state_position_x[-1] = traci.vehicle.getPosition(self.rl_car_id)[0]
            state_position_y[-1] = traci.vehicle.getPosition(self.rl_car_id)[1]
            # state_speed = np.clip(state_speed, 0, self.maxSpeed)
            state_latSpeed[-1] = traci.vehicle.getLateralSpeed(self.rl_car_id)
            state_acc[-1] = traci.vehicle.getAcceleration(self.rl_car_id)
            # state_position_x = np.clip(state_position_x, -abs(self.observation_space['xPos'].low),
            #        abs(self.observation_space['xPos'].high))
            # state_position_y = np.clip(state_position_y, -abs(self.observation_space['yPos'].low),
            #                          abs(self.observation_space['yPos'].high))

        sigmavalue = 1


        state_speed = state_speed + np.random.normal(0, 0.277778,np.size(state_position_x)) * noise_level 

        state_position_x = state_position_x + np.random.normal(0, 1.5,np.size(state_position_x)) * noise_level 

        state_position_y = state_position_y + np.random.normal(0, 1.5,np.size(state_position_x)) * noise_level 

        state_acc = state_acc + np.random.normal(0, 0.3,np.size(state_position_x)) * noise_level 


        state_latSpeed =  state_latSpeed + np.random.normal(0, 0.277778, np.size(state_position_x)) * noise_level 

        state = {
            'image': state_image.astype(np.uint8),
            'xPos': np.array(state_position_x, dtype=np.float32),
            'yPos': np.array(state_position_y, dtype=np.float32),
            'velocity': np.array(state_speed, dtype=np.float32),
            'latSpeed': np.array(state_latSpeed, dtype=np.float32),
            'acceleration': np.array(state_acc, dtype=np.float32),
        }
        return state

class BSM_Perfect_Noise_Image(BsmMergeAllRewards):
    def getObservations(self):
        # returns observations of the state

        state_speed = np.ones(7) * self.maxSpeed
        state_position_x = np.ones(7)
        state_position_y = np.ones(7)

        state_latSpeed = np.zeros(7)
        state_acc = np.zeros(7)
        vehicle_ids = self.getVehicleIds()
        state_image = np.array(self.render())

        if vehicle_ids:
            for vehicle in vehicle_ids:
                if not "rl" in vehicle:
                    traci.vehicle.setColor(vehicle, color=(255, 255, 255, 255))  # change vehicle color to white

            obsLane0, obsLane1 = self.getobservedVehicles(vehicle_ids)

            for i, vehicle in enumerate(obsLane0):
                maxSpeed = traci.vehicle.getMaxSpeed(vehicle_ids[0])
                # print(vehicle)
                if vehicle:
                    if vehicle[0] not in ["no_vehicle", "", None]:
                        state_speed[i] = traci.vehicle.getSpeed(vehicle[0])
                        state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                        state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
                        state_latSpeed[i] = traci.vehicle.getLateralSpeed(vehicle[0])
                        state_acc[i] = traci.vehicle.getAcceleration(vehicle[0])
                # traci.vehicle.setColor(vehicle[0], color=(255, 0, 255, 255))  # change vehicle color to blue

            for i, vehicle in enumerate(obsLane1, len(obsLane0)):
                # print(vehicle)
                if vehicle:

                    if vehicle[0] not in ["no_vehicle", "", None]:
                        state_speed[i] = traci.vehicle.getSpeed(vehicle[0])

                        state_acc[i] = traci.vehicle.getAcceleration(vehicle[0])
                        state_position_x[i] = traci.vehicle.getPosition(vehicle[0])[0]
                        state_latSpeed[i] = traci.vehicle.getLateralSpeed(vehicle[0])
                        state_position_y[i] = traci.vehicle.getPosition(vehicle[0])[1]
                # traci.vehicle.setColor(vehicle[0], color=(255, 255, 0, 255))  # change vehicle color to blue

            # rl state information
            state_speed[-1] = traci.vehicle.getSpeed(self.rl_car_id)  # / self.maxSpeed
            state_position_x[-1] = traci.vehicle.getPosition(self.rl_car_id)[0]
            state_position_y[-1] = traci.vehicle.getPosition(self.rl_car_id)[1]
            # state_speed = np.clip(state_speed, 0, self.maxSpeed)
            state_latSpeed[-1] = traci.vehicle.getLateralSpeed(self.rl_car_id)
            state_acc[-1] = traci.vehicle.getAcceleration(self.rl_car_id)
            # state_position_x = np.clip(state_position_x, -abs(self.observation_space['xPos'].low),
            #        abs(self.observation_space['xPos'].high))
            # state_position_y = np.clip(state_position_y, -abs(self.observation_space['yPos'].low),
            #                          abs(self.observation_space['yPos'].high))

        sigmavalue = 1
        state_image = gaussian_filter(state_image, sigma=sigmavalue)

        state = {
            'image': state_image.astype(np.uint8),
            'xPos': np.array(state_position_x, dtype=np.float32),
            'yPos': np.array(state_position_y, dtype=np.float32),
            'velocity': np.array(state_speed, dtype=np.float32),
            'latSpeed': np.array(state_latSpeed, dtype=np.float32),
            'acceleration': np.array(state_acc, dtype=np.float32), 
        }
        return state

