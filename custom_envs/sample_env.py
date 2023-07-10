import gym
import random
from gym.spaces import Discrete, Box
import numpy as np
from gym import Env
gym.logger.set_level(40)
class customEnv(Env):
    ''' four key functions
    init
    step
    random
    reset
    '''
    # env for shower temprature https://www.youtube.com/watch?v=bD6V3rcr_54
    def __init__(self):
        # action space for example down , stay , up
        self.actionSpace = Discrete(3)
        # oversation space allows us to have continous values over the ranfe of layers
        self.observationSpace = Box(low= np.array([0]),high=np.array([100]), )

        #state is the information from the env

        self.state = 38 - random.randint(-3,3)
        self.episodeLength = 60 # secs

    def step(self, action):
        # how we rake the action

        # apply action
        # here the actions are 0,1 ,2 to reduce the temp, stay the temp and increase it
        # if 0 : state = state + 0 -1 to reduce the state by 1
        # if 1 : state = state + 1-1 to keep the same state
        # if 2 : state = state + 2-1 to increaset the temp by 1

        self.state += action -1

        # reduce the episode length
        self.episodeLength -=1 #sec

        # calc reward
        # the aim is that the temp remains between 17 and 39 so we give a reward if it is in this temp
        # else we give a negative reward

        if self.state >= 17 and self.state<= 39 :
            reward = 1
        else:
            reward = -1

        # check if shower is done
        if self.episodeLength == 0:    done = True
        else:   done = False

        # some random noise to the state
        self.state += random.randint(-1,1)

        info ={}

        # how open ai requires
        return self.state, reward, done, info

    def render(self):
        # if to visualize
        pass
    def reset(self):
        self.state  = 38 - random.randint(-3,3)
        self.episodeLength = 60
        return self.state
env = customEnv()

for episode in range(1,100):
    env.reset()
    done= False
    score = 0
    while not done:
        env.render()
        action = env.actionSpace.sample()
        state_, reward, done, info = env.step(action)

        score += reward
    print(f'Episode {episode} score {score} ')
