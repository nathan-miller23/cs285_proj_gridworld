from gym_minigrid.minigrid import TILE_PIXELS, COLORS
from gym_minigrid.rendering import highlight_img
from gym_minigrid.envs.mygridworld import MyEnv
from gym_minigrid.wrappers import *
import gym_minigrid.window

import numpy as np

def show_grid_gradient(env, matrix_vals, highligh_color='red', tile_size=TILE_PIXELS, scale=(0, 10), alpha=0.75):
    low, high = scale
    img = env.grid.render(tile_size=TILE_PIXELS)
    matrix_vals = (matrix_vals - low) / (high - low)
    color_arr = COLORS[highligh_color]

    if not matrix_vals.shape == (env.width - 2, env.height - 2):
        raise ValueError("Expected matrix_vals shape {} but got {}".format((env.width - 2, env.height - 2), matrix_vals.shape))

    for i in range(1, env.width-1):
        for j in range(1, env.height-1):
            ymin = j * tile_size
            ymax = (j+1) * tile_size
            xmin = i * tile_size
            xmax = (i+1) * tile_size
            tile = img[ymin:ymax, xmin:xmax, :]
            tile_val = matrix_vals[i-1, j-1]
            highlight_img(tile, color_arr * tile_val, alpha=alpha)

    
    window = gym_minigrid.window.Window('MyMinigrid Visualization')
    window.show_img(img)
    window.show()
    return img

if __name__ == '__main__':
    env = MyEnv(size=9, good_goal_pos=(7, 8), bad_goal_pos=(8, 7))
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    mat_vals = np.ones((env.width - 2, env.height - 2)) * 10
    show_grid_gradient(env, mat_vals, scale=(0, 20))
