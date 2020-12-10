from gym_minigrid.envs.mygridworld import MyEnv
from gym_minigrid.wrappers import *
from agents.hardcoded_agents import *
import numpy as np
from collections import Counter

env = MyEnv(size=9, good_goal_pos=(7, 8), bad_goal_pos=(8, 7), max_steps=int(1e3))
env = FullyObsWrapper(env)
env = ImgObsWrapper(env)

agent = GoToGoodGoalAgent(observation_space=env.observation_space, action_space=env.action_space, epsilon=0.5)
# agent = RandGoodGoalAgent(observation_space=env.observation_space, action_space=env.action_space)
# agent = RandAgent(observation_space=env.observation_space, action_space=env.action_space)
num_episodes = 500


rewards = []
for _ in range(num_episodes):
    obs = env.reset()
    done = False
    reward = 0.0
    while not done:
        action = agent.action(obs)
        obs, curr_reward, done, _ = env.step(action)
        reward += curr_reward
    rewards.append(reward)

c = Counter()
c.update(rewards)
print("counts", c)
print("reward mean", np.mean(rewards))
print("reward std", np.std(rewards))
