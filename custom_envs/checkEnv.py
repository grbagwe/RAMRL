from ramp_env3 import SumoRampEnv
import os
from gym.wrappers.rescale_action import RescaleAction
from gym.wrappers.resize_observation import
# simpath = "/home/gauravb/Documents/MichiganTech/Programming/CustomRampTraining/custom_envs/sumo_Config/ramp_1.sumocfg"
# simpath = os.getcwd()+"/custom_envs/sumo_Config/ramp_1.sumocfg"
from gym.spaces import Box
env = SumoRampEnv()
min_action = -1
max_action = +1
print('before \n ', env.action_space.high,'high', env.action_space.low ,'low')
env  = RescaleAction(env, min_action, max_action)
for i in range(0,100):
    print(env.action_space.sample())

print('after \n ', env.action_space.high,'high', env.action_space.low ,'low')

from gym.utils.env_checker import check_env

check_env(env)
#
# for episode in range(1,10):
#     env.reset()
#     done= False
#     score = 0
#     while not done:
#         env.render()
#         action = +30 #env.action_space.sample()
#         state_, reward, done, info = env.step(action)
#
#         score += reward
#     print(f'Episode {episode} score {score} ')
