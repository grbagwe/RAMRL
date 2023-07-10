import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecVideoRecorder, VecMonitor, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from gym.wrappers.rescale_action import RescaleAction
# from custom_envs.rampTaperEnv_half import SumoRamp
from gym.spaces import Box
# from custom_envs.bsmMerge import BsmMergeAllRewards as BsmMerge
# from custom_envs.bsmMerge import BsmMerge
# load simple cnn + bsm reward env
# from custom_envs.MultiMerge import MultiMerge
# load cnn + bsm all rewards env
from custom_envs.MultiMergeParallel import MultiMergeAllRewards as MultiMerge


import os
import wandb, glob
#from customFeatureExtractor import CustomCombinedExtractor, CustomNatureCNN
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
import argparse

parser = argparse.ArgumentParser(description='test PPO multi model')

parser.add_argument("dir", help="model path")
parser.add_argument("--render", default =0, help = "should render default 0")

parser.add_argument("stats_path", help="vec norm stats path  file")
parser.add_argument("config", help="Config file")
args = parser.parse_args()

module = __import__("config_file",fromlist= [args.config])
exp_config = getattr(module, args.config)


timesteps = 3e6
sub_timesteps = 10000

config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": timesteps,
    "env_name": "SumoRamp()",
    "sub_timesteps": sub_timesteps
}


pdir = os.path.abspath('../')
dir = os.path.join(pdir, 'SBRampSavedFiles/wandbsavedfiles')


# env = SumoRamp(action_space=action_space, obsspaces=obsspaces, sumoParameters = sumoParameters, weights= weights, isBaseline=False)
policy_kwargs = exp_config.policy_kwargs

action_space = exp_config.action_space

image_shape = exp_config.image_shape
obsspaces = exp_config.obsspaces

weights = exp_config.weights
sumoParameters = exp_config.sumoParameters

min_action = -1
max_action = +1

video_folder = dir + '/logs/videos/'
video_length = 600

def make_env(env_id, rank, seed=0, monitor_dir = None):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():


        env = MultiMerge(action_space=action_space, obsspaces=obsspaces, sumoParameters=sumoParameters, weights=weights,
                       isBaseline=False,render=0)
        env.seed(seed + rank)
        env = RescaleAction(env, min_action, max_action)
        monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
        if monitor_path is not None:
            os.makedirs(monitor_dir, exist_ok=True)
        env = Monitor(env, filename=monitor_path)

        return env

    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    run = wandb.init(
        project="Multi_Testing",
        dir=dir,
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        magic=True
    )

    env_id = "MultiMerge"
    num_cpu = 1  # Number of processes to use
    # Create the vectorized environment
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
        # env = VecVideoRecorder(env, video_folder=f"{dir}/videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0,
    #                        video_length=300)

    # add vstack
    env = VecFrameStack(env, n_stack=4) # stack 4 frames
    env = VecNormalize.load(args.stats_path,env)
    env = VecMonitor(venv=env)


    #code = wandb.Artifact('project-source', type='code')
    #for path in glob.glob('**/*.py', recursive=True):
    #    code.add_file(path)

    #wandb.run.use_artifact(code)



    model = PPO.load(args.dir, env)

    obs = env.reset()
    n_games = 300
    for i_games in range(n_games):

        done = False
        obs = env.reset()
        score = 0
        num_collisions = 0
        mergeTime = 0

        while not done:
            action, _states = model.predict(obs)
            print('action', action)
            obs, rewards, done, info = env.step(action)
            if int(args.render)==1:
                env.render()
            score += rewards

            print('rewards', rewards)
            if int(info[0]['terminal']) == -1:
                num_collisions += 1
            if int(info[0]['terminal']) != 0:
                mergeTime = int(info[0]['mergeTime'])
        print(f"score {score} num_collisions : {num_collisions} , mergetime : {mergeTime}")
        wandb.log({
            "episodic score": score,
            "num_collisions": num_collisions,
            "mergeTime": mergeTime
        }, step=i_games)
