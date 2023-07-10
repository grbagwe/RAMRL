import numpy as np
# from ppo4_torch import Agent
# from custom_envs.ramp_baseline import SumoRampEnv
from csv import writer
import datetime
# from torchvision import transforms
import os
from torch.utils.tensorboard import SummaryWriter
import wandb
from custom_envs.rampTaperEnv_2 import SumoRamp
from gym.spaces import Box
import numpy as np
from gym.wrappers.rescale_action import RescaleAction

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
           'alphasl1' :0.2,
           'rSuccess' :100,
           'alphaO' : 0.03,
           'rTimeAlpha': 0.001,
           'alphaD' : 0.001,
           'rC':-250,
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
env = SumoRamp(action_space=action_space, obsspaces=obsspaces, sumoParameters = sumoParameters, weights= weights, isBaseline=True)
env = RescaleAction(env, min_action, max_action)
# to_tensor = transforms.ToTensor()

x = datetime.datetime.now()
dir = '../CustomRampSavedFiles/baselines/' # + 'experimentscore' + '_' + str(x.month) + '_' + str(x.day) + '_' + str(x.hour) + '_' + str(    x.minute)

# dir = "/home/grbagwe/Programming/SBRampSavedFiles/wandbsavedfiles"
run = wandb.init(
    project="baselines",
    dir=dir,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
    magic=True
)

if not os.path.exists(dir):
    os.makedirs(dir)
torchwriter = SummaryWriter(log_dir=dir)
print('dir \n\n\n', dir)
print('os.getpid() \n\n\n', os.getpid())
scoreFile = dir + '/progress.csv'

List = ['episode', 'score', 'avg score',
        'time_steps', 'learning_steps', 'num_collisions']
with open(scoreFile, 'a') as write_scores:
    writer_object = writer(write_scores)
    writer_object.writerow(List)
    write_scores.close()
BufferSize = 256
min_batch_size = 64

obs_space = env.observation_space.sample()
# agent = Agent(n_actions=env.action_space.shape[0], batch_size=batch_size,
#               alpha= alpha, n_epochs=n_epochs,
#               image_dims=obs_space['image'].shape, velocity_dims=obs_space['velocity'].shape,
#               positionX_dims=obs_space['xPos'].shape, positionY_dims=obs_space['yPos'].shape,
#               # maxAcc= env.action_space.high[0],
#               # minAcc= env.action_space.low[0]
#               )
# agent = Agent(image_size=obs_space['image'].shape, v_size=obs_space['velocity'].shape, x_size=obs_space['xPos'].shape,
#               y_size=obs_space['yPos'].shape, action_size=env.action_space.shape[0],
#               minActionValue=env.action_space.low[0], maxActionValue=env.action_space.high[0],
#               min_batch_size=min_batch_size, ppo_epsilon=0.2,
#               lr_alpha=1e-4, gamma=0.99, gae_lambda=0.95, ppo_epochs=10)
n_games = 300
print('min max action value \n\n\n', env.action_space.high[0], env.action_space.low)

figure_file = 'plots/sumo.png'

total_reward = 0
total_steps = 0
n_steps = 0
learn_iter = 0
score_history = []
best_score = 0
mergetime =[]
for i_ep in range(n_games):
    score = 0
    state = env.reset()
    done = False
    num_collisions = 0

    while not done:
        # action, prob, value = agent.choose_action(state)
        action = [1]
        state_, reward, done, info = env.step(action)
        n_steps += 1
        score += reward

        state2 = state.copy()
        # image_reshaped = to_tensor(state2['image']).unsqueeze(0).numpy()
        num_collisions += info['collision']
        torchwriter.add_scalar('intermediate rewards', reward, n_steps)

        if 'mergeTime' in info.keys():
            mergetime.append(info['mergeTime'])
        if done:
            print('done', done)

        # agent.remember(image_reshaped, state2['velocity'], state2['xPos'], state2['yPos'], action, prob, value, reward,
        #                done)
        # # when the buffer reaches the size update the model
        # if n_steps % BufferSize == 0:
        #     agent.learn()
        #     learn_iter += 1
    score_history.append(score)
    meanMergeTime = np.mean(mergetime)
    avg_score = np.mean(score_history[-100:])
    if avg_score > best_score:
        best_score = avg_score
    saveScores = [i_ep, score, avg_score, n_steps, learn_iter, num_collisions]
    with open(scoreFile, 'a') as write_scores:
        writer_object = writer(write_scores)
        writer_object.writerow(saveScores)
        write_scores.close()
    wandb.log({"score": score, "epoch": i_ep,
           "meanMergeTime": meanMergeTime,
           "avg_score":avg_score,
           "number of collisions": num_collisions}, step=i_ep)
    torchwriter.add_scalar('episode/score',score, i_ep)
    torchwriter.add_scalar('episode/meanMergeTime',meanMergeTime, i_ep)
    torchwriter.add_scalar('episode/avg_score', avg_score, i_ep)
    torchwriter.add_scalar('episode/number of collisions', num_collisions, i_ep)


    print('episode', i_ep, 'score %.1f' % score, 'avg score %.1f' % avg_score,
          'time_steps', n_steps, 'learning_steps', learn_iter, "number of collisions ", num_collisions)

