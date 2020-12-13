from gym_minigrid.minigrid import TILE_PIXELS, COLORS
from gym_minigrid.rendering import highlight_img
from gym_minigrid.envs.mygridworld import MyEnv
from gym_minigrid.wrappers import *
from utils import load
from rl import tabular_learning, get_value_table_from_states, get_initial_value_table, encode, get_value_table_from_obs
from agents import GoToGoodGoalAgent, AgentFromTorchEnsemble
from train import LOG_DIR, Net
from generate_data import DATA_DIR
import gym_minigrid.window
import matplotlib.cm as color
import argparse, os, torch


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
    parser.add_argument("--type", "-t", choices=["astrat", "values", "freq", "uncertainty"], default="astrat")
    parser.add_argument("--agent_checkpoints", '-cpts', nargs='+', type=str, default=LOG_DIR)
    parser.add_argument("--data_dir", "-dp", type=str, default=DATA_DIR)
    parser.add_argument('--infile_name', '-i', type=str, default="my_exp")
    params = vars(parser.parse_args())

    exp_dir = os.path.join(params['data_dir'], params['infile_name'])
    env_load_loc, agent_load_loc = os.path.join(exp_dir, 'env.pkl'), os.path.join(exp_dir, 'agent.pkl')
    data_load_loc = os.path.join(exp_dir, 'data.pkl')

    mat_vals = None
    scale = None
    if params['type'] == 'astrat':
        env, agent = load(env_load_loc), load(agent_load_loc)
        _, _, A_strat = tabular_learning(env, agent, gamma=0.95, state_func=True)
        mat_vals = get_value_table_from_states(env, A_strat)
        A_strat_max = env.good_goal_reward - env.bad_goal_reward
        scale = (0, A_strat_max)
    elif params['type'] == 'value':
        env, agent = load(env_load_loc), load(agent_load_loc)
        v_pi, _, _ = tabular_learning(env, agent, gamma=0.95, state_func=True)
        mat_vals = get_value_table_from_states(env, v_pi)
        v_min = env.bad_goal_reward
        v_max = env.good_goal_reward
        scale = (v_min, v_max)
    elif params['type'] == 'freq':
        data = load(data_load_loc)['states']
        mat_vals = get_freq_table_from_data(data, env)
        scale = None
    else:
        ensemble = []
        for checkpoint in params['agent_checkpoints']:
            state_dict, params = torch.load(checkpoint, map_location=torch.device('cpu'))
            model = Net(**params)
            model.load_state_dict(state_dict)
            ensemble.append(model)
        agent = AgentFromTorchEnsemble(ensemble)
        env = load(env_load_loc)
        mat_vals = get_value_table_from_obs(env, agent.uncertainty)

    show_grid_gradient(env, mat_vals, scale=scale)
    