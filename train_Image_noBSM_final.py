import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecVideoRecorder, VecMonitor, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from gym.wrappers.rescale_action import RescaleAction
from gym.spaces import Box
from custom_envs.MultiMerge import Image_No_BSM as MultiMerge

import os
import wandb, glob
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
import argparse

parser = argparse.ArgumentParser(description='train PPO multi model')
parser.add_argument("config", help="Config file")
parser.add_argument("--noise_sigma",default=0.1 ,  help="Image noise sigma value")

args = parser.parse_args()
module = __import__("config_file",fromlist= [args.config])
exp_config = getattr(module, args.config)

timesteps = 500000

config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": timesteps,
    "env_name": "SumoRamp()",
}
pdir = os.path.abspath('../')
dir = os.path.join(pdir, 'SBRampSavedFiles/wandbsavedfiles')

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
        return env
    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    run = wandb.init(
            project="RMMRL-Training",
        name=f"Image+NoBSM_{args.config}",
        dir=dir,
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        magic=True
    )

    env_id = "MultiMerge"
    num_cpu =16# Number of processes to use
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    env = VecFrameStack(env, n_stack=4) # stack 4 frames
    env = VecNormalize(env, norm_obs=True, norm_reward=True, training=True)
    env = VecMonitor(venv=env)
    model = PPO(config["policy_type"], 
            env,
            verbose=3,
            policy_kwargs=policy_kwargs,
            gamma=0.99,
            n_steps=512,
            learning_rate=0.0003,
            vf_coef=0.042202,
            max_grad_norm=0.9,
            gae_lambda=0.95,
            n_epochs=10,
            clip_range=0.2,
            batch_size=256, 
            tensorboard_log=f"{dir}"
            )

    model.learn(
                total_timesteps=int(config["total_timesteps"]),
                callback=WandbCallback(
                gradient_save_freq=5,
                model_save_freq=5000,
                model_save_path=f"{dir}/models/{run.id}",
                verbose=2,
                ), )
    stats_path = os.path.join(f"{dir}/models/{run.id}/", "vec_normalize.pkl")
    env.save(stats_path)

