from gym_minigrid.envs.mygridworld import MyEnv
from utils import load, DATA_DIR
from rl import encode
from gym_minigrid.wrappers import *
from agents.hardcoded_agents import *
import numpy as np
import os

# data = load(os.path.join(DATA_DIR, 'rbg_test', 'data.pkl'))

# print(data.keys())
# for key, val in data.items():
#     print("{} shape: {}".format(key, val.shape))

env = MyEnv(size=9, good_goal_pos=(7, 8), bad_goal_pos=(8, 7), max_steps=int(1e3))
env = FullyObsWrapper(env)
env = RGBImgObsWrapper(env)
env = ImgObsWrapper(env)




