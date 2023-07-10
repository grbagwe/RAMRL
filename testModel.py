import numpy as np
# from ppo4_torch import Agent
import wandb

from custom_envs.rampTaperEnv_2 import SumoRamp
from csv import writer
import datetime
# from torchvision import transforms
import os
from torch.utils.tensorboard import SummaryWriter
import cv2

from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from customFeatureExtractor import CustomCombinedExtractor
from gym.wrappers.rescale_action import RescaleAction
from gym.spaces import Box

# env_id = 'CartPole-v1'

video_length = 3000
min_action = -1
max_action = +1
timesteps = 3e6
model_save_freq = 10000

x = datetime.datetime.now()
pdir = os.path.abspath('../')
dir = os.path.join(pdir, 'SBRampSavedFiles/TestingRamp/testimagesl3aej15f')
# dir ="/home/grbagwe/Programming/SBRampSavedFiles/TestingRamp/testimagesl3awj15f/"
# fir = /home/grbagwe/Programming/SBRampSavedFiles/wandbsavedfiles/wandb/run-20220426_212441-l3aej15f/files"
# if not os.path.exists(dir):
#     os.mkdir(dir)

run = wandb.init(
    project="TestingCNNRamp",
    dir=dir,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
    magic=True
)

torchwriter = SummaryWriter(log_dir=dir)
print('dir \n\n\n', dir)
print('os.getpid() \n\n\n', os.getpid())
video_folder = dir + '/videos/'
scoreFile = dir + '/progress.csv'

List = ['episode', 'score', 'avg score',
        'time_steps', 'learning_steps', 'num_collisions']
with open(scoreFile, 'a') as write_scores:
    writer_object = writer(write_scores)
    writer_object.writerow(List)
    write_scores.close()

action_space = {'high': 3,
                'low': -4.5}
image_shape = (200, 768)
obsspaces = {
    'image': Box(low=0, high=255, dtype=np.uint8, shape=(image_shape[0], image_shape[1], 3)),
    'velocity': Box(low=0, high=70, shape=(7,)),
    'xPos': Box(low=-100000, high=100000, shape=(7,)),
    'yPos': Box(low=-100000, high=100000, shape=(7,)),
}

weights = {'alphasl0': 0.7,
           'alphasl1': 0.2,
           'rSuccess': 100,
           'alphaO': 0.03,
           'rTimeAlpha': 0.001,
           'alphaD': 0.001,
           'rC': -250,
           'alphaDistance': 0.02,
           'alphaP': 0.01,
           'alphaJ': 0.5
           }
sumoParameters = {'maxSpeed': 55,
                  'episodeLength': 600
                  }

min_action = -1
max_action = +1
timesteps = 3e6

# config = {
#    "policy_type": "MultiInputPolicy",
#    "total_timesteps": timesteps,
#    "env_name": SumoRamp(),
# }
# def make_env():
#    env = config["env_name"]
#    env = RescaleAction(env, min_action, max_action)
#
#    # env = Monitor(env)  # record stats such as returns
#    return env


# env = make_env()

env = SumoRamp(action_space=action_space, obsspaces=obsspaces, sumoParameters=sumoParameters, weights=weights,
               isBaseline=False)
env = RescaleAction(env, min_action, max_action)

policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    features_extractor_kwargs=dict(cnn_output_dim=2046),
    net_arch=[1024, dict(vf=[512, 128, 32], pi=[512, 128, 32])],
)

#
# model = PPO("MultiInputPolicy", env,
#             policy_kwargs=policy_kwargs,
#             verbose=1)
# model = PPO("MultiInputPolicy", env)

# model.load('../SBRampSavedFiles/SBTaperModel.zip')

# model = PPO.load(os.path.join(pdir,'SBRampSavedFiles/models/l3aej15f/model'), env)

model = PPO.load(os.path.join(pdir,'../trainedSBModels/rural-snow/model'), env)

obs_space = env.observation_space.sample()

n_games = 300
print('min max action value \n\n\n', env.action_space.high[0], env.action_space.low)

total_reward = 0
total_steps = 0
n_steps = 0
learn_iter = 0
score_history = []
best_score = 0
mergetime = []
for i_ep in range(n_games):
    score = 0
    state = env.reset()
    done = False
    num_collisions = 0

    while not done:
        action, _ = model.predict(state)
        # action = env.action_space.sample()
        print('action', action)
        state_, reward, done, info = env.step(action)
        print('reward', reward)
        n_steps += 1
        score += reward

        state2 = state_.copy()
        # image_reshaped = to_tensor(state2['image']).unsqueeze(0).numpy()
        num_collisions += info['collision']
        torchwriter.add_scalar('intermediate rewards', reward, n_steps)
        state = state_

        if 'mergeTime' in info.keys():
            mergetime.append(info['mergeTime'])
        imageStr = dir + str(n_games) + '_' + str(n_steps) + '.png'
        print(imageStr)
        # print('info keys',info.keys())
        cv2.imwrite(imageStr, info['frame'])

    score_history.append(score)
    meanMergeTime = np.mean(mergetime)
    avg_score = np.mean(score_history[-100:])
    if avg_score > best_score:
        best_score = avg_score
    saveScores = [i_ep, score, avg_score, n_steps, learn_iter, num_collisions]
    wandb.log({"score": score, "epoch": i_ep,
               "meanMergeTime": meanMergeTime,
               "avg_score": avg_score,
               "number of collisions": num_collisions}, step=i_ep)

    with open(scoreFile, 'a') as write_scores:
        writer_object = writer(write_scores)
        writer_object.writerow(saveScores)
        write_scores.close()

    torchwriter.add_scalar('episode/score', score, i_ep)
    torchwriter.add_scalar('episode/meanMergeTime', meanMergeTime, i_ep)
    torchwriter.add_scalar('episode/avg_score', avg_score, i_ep)
    torchwriter.add_scalar('episode/number of collisions', num_collisions, i_ep)

    print('episode', i_ep, 'score %.1f' % score, 'avg score %.1f' % avg_score,
          'time_steps', n_steps, 'learning_steps', learn_iter, "number of collisions ", num_collisions)
