from gym_minigrid.envs.tightrope import TightRopeEnv
from gym_minigrid.envs.randompoints import RandomPoints
from gym_minigrid.envs.mygridworld import MyEnv
from gym_minigrid.minigrid import Direction
import argparse

SUPPORTED_ENVS = ['tight_rope', 'random', 'my_env']

def _get_env(env_name):
    assert env_name in SUPPORTED_ENVS, "{} environment not supported!".format(env_name)

    if env_name == 'tight_rope':
        return TightRopeEnv()
    if env_name == 'random':
        return RandomPoints()
    if env_name == 'my_env':
        return MyEnv()
    return None

def main(env_name):
    env = _get_env(env_name)
    while True:
        _ = env.reset()
        done = False
        while not done:
            env.render()
            key_input = input("action: ")
            if key_input == "s":
                action = Direction.down
            elif key_input == "a":
                action = Direction.left
            elif key_input == "w":
                action = Direction.up
            elif key_input == "d":
                action = Direction.right
            else:
                continue

            _, _, done, _ = env.step(action)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '-e', choices=SUPPORTED_ENVS, default='my_env')


    vars = vars(parser.parse_args())
    main(**vars)