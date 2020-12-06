from gym_minigrid.minigrid import TILE_PIXELS, COLORS
from gym_minigrid.rendering import highlight_img
from gym_minigrid.envs.mygridworld import MyEnv
from gym_minigrid.wrappers import *
from utils import load
from rl import tabular_learning, get_value_table_from_states
from agents import GoToGoodGoalAgent
import gym_minigrid.window
import matplotlib.cm as color

import numpy as np

def show_grid_gradient(env, matrix_vals, colormap='Reds', tile_size=TILE_PIXELS, scale=(0, 10), alpha=0.75, title="My Visualization"):
    if not matrix_vals.shape == (env.width, env.height):
        raise ValueError("Expected matrix_vals shape {} but got {}".format((env.width, env.height), matrix_vals.shape))
    
    if scale:
        low, high = scale
    else:
        low, high = np.min(matrix_vals[1:-1, 1:-1]), np.max(matrix_vals[1:-1, 1:-1])

    img = env.grid.render(tile_size=TILE_PIXELS)
    matrix_vals = (matrix_vals - low) / (high - low)
    my_cmap = color.get_cmap(colormap)
    color_array = np.zeros((env.width, env.height, 4))
    color_array[1:-1, 1:-1, :] = my_cmap(matrix_vals[1:-1, 1:-1])

    for i in range(1, env.width-1):
        for j in range(1, env.height-1):
            ymin = j * tile_size
            ymax = (j+1) * tile_size
            xmin = i * tile_size
            xmax = (i+1) * tile_size
            tile = img[ymin:ymax, xmin:xmax, :]
            highligh_color = color_array[i, j, :-1] * 255
            highlight_img(tile, highligh_color, alpha=alpha)

    
    window = gym_minigrid.window.Window(title)
    window.show_img(img)
    window.show()
    return img

if __name__ == '__main__':
    env = load("exp_env.pkl")
    agent = GoToGoodGoalAgent(action_space=env.action_space, observation_space=env.observation_space, epsilon=0.3)
    v_pi, _, A_strat = tabular_learning(env, agent, gamma=0.9, state_func=True)
    mat_vals = get_value_table_from_states(env, v_pi)
    A_strat_max = env.good_goal_reward - env.bad_goal_reward
    v_max = env.good_goal_reward
    v_min = env.bad_goal_reward
    show_grid_gradient(env, mat_vals, scale=None)
    