from gym_minigrid.envs.mygridworld import MyEnv
from gym_minigrid.wrappers import *
from agents.hardcoded_agents import GoToGoodGoalAgent

env = MyEnv(size=9, good_goal_pos=(7, 8), bad_goal_pos=(8, 7))
env = FullyObsWrapper(env)
env = ImgObsWrapper(env)

agent = GoToGoodGoalAgent(observation_space=env.observation_space, action_space=env.action_space, epsilon=0.3)
obs = env.reset()
done = False
while not done:
    env.render()
    action = agent.action(obs)
    obs, _, done, _ = env.step(action)
