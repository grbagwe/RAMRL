import ray
from ramp_env3 import SumoRampEnv
from ray.rllib.agents.ppo import PPOTrainer


import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print


''''
https://docs.ray.io/en/latest/rllib/rllib-models.html

the vision network case, you’ll probably have to configure conv_filters, if your environment observations have custom 
sizes. For example, "model": {"dim": 42, "conv_filters": [[16, [4, 4], 2], [32, [4, 4], 2], [512, [11, 11], 1]]} for
 42x42 observations. Thereby, always make sure that the last Conv2D output has an output shape of [B, 1, 1, X] 
 ([B, X, 1, 1] for PyTorch), where B=batch and X=last Conv2D layer’s number of filters, so that RLlib can flatten it.
  An informative error will be thrown if this is not the case.
'''
config = ppo.DEFAULT_CONFIG.copy()

config = {
    # this is a dict
    # "env": SumoRampEnv,
    "num_workers": 1,
    #     "framework" : "tf2",
    "num_gpus": 1,
    "model": {
        "dim": 512,
        "conv_filters": [  # [[16, [4, 4], 2], [32, [4, 4], 2], [512, [11, 11], 1], [1000, 1, 512]],#, [1000,512, 1]],
            [96, 11, 4],  # 126
            [256, 5, 2],  # 61
            [384, 3, 2],  # 30
            [384, 3, 2],  # 14
            [256, 3, 2],  # 6
            [256, 3, 2],  # 2
            [256, 1, 128],

        ],  # lenet
        "post_fcnet_hiddens": [256, 256],
        #         "post_fcnet_activation": "relu",
        #         "fcnet_hiddens" : [10, 10 ],
        #         "fcnet_activation" : "relu",

    },
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": True,
    }

}
#
# from ray import tune
#
# def tune_func(config):
#     tune.util.wait_for_gpu()
#     train()
#
# tune.run(PPOTrainer, config=config, verbose=3,
#          # resources_per_trial={"cpu": 12, "gpu": 1} ,
#          reuse_actors=True,
#          stop={"training_iteration": 10e3})





ray.init()


trainer = ppo.PPOTrainer(config=config, env=SumoRampEnv)

# Can optionally call trainer.restore(path) to load a checkpoint.

for i in range(1000):
    # Perform one iteration of training the policy with PPO
    result = trainer.train()
    print(pretty_print(result))

    if i == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
    ray.shutdown()

