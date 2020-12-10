from gym_minigrid.minigrid import TILE_PIXELS, COLORS
from gym_minigrid.rendering import highlight_img
from gym_minigrid.envs.mygridworld import MyEnv
from gym_minigrid.wrappers import *
from utils import load
from rl import tabular_learning, get_value_table_from_states, get_initial_value_table, encode
from agents import GoToGoodGoalAgent
import gym_minigrid.window
import matplotlib.cm as color
import argparse, os

import numpy as np

CURR_DIR = os.path.abspath('.')

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

def get_freq_table_from_data(states, env):
    table = get_initial_value_table(env)
    N = len(states)

    for state in states:
        table[encode(state)] += 1 / N

    freq_func = lambda state : table[state]
    return get_value_table_from_states(env, freq_func)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", "-t", choices=["astrat", "values", "freq"], default="astrat")
    parser.add_argument("--data_path", "-dp", type=str, default=os.path.join(CURR_DIR, 'data.pkl'))
    parser.add_argument('--agent_path', '-ap', type=str, default=os.path.join(CURR_DIR, 'agent.pkl'))
    parser.add_argument('--env_path', '-ep', type=str, default=os.path.join(CURR_DIR, 'env.pkl'))

    params = vars(parser.parse_args())
    env = load(params['env_path'])
    agent = load(params['agent_path'])

    mat_vals = None
    scale = None
    if params['type'] == 'astrat':
        _, _, A_strat = tabular_learning(env, agent, gamma=0.95, state_func=True)
        mat_vals = get_value_table_from_states(env, A_strat)
        A_strat_max = env.good_goal_reward - env.bad_goal_reward
        scale = (0, A_strat_max)
    elif params['type'] == 'value':
        v_pi, _, _ = tabular_learning(env, agent, gamma=0.95, state_func=True)
        mat_vals = get_value_table_from_states(env, v_pi)
        v_min = env.bad_goal_reward
        v_max = env.good_goal_reward
        scale = (v_min, v_max)
    else:
        data = load(params['data_path'])['states']
        mat_vals = get_freq_table_from_data(data, env)
        scale = None

    show_grid_gradient(env, mat_vals, scale=scale)
    