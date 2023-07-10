import gym
env = gym.make('Hopper-v2')

env.reset()

for _ in range(100):
    env.render()
    env.step(env.action_space.sample())
env.close()
