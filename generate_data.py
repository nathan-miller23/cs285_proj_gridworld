import os, sys, argparse, pickle
import numpy as np

from gym_minigrid.envs.mygridworld import MyEnv
from gym_minigrid.wrappers import *
from agents.hardcoded_agents import GoToGoodGoalAgent

CURR_DIR = os.path.abspath('.')

def save(data, outfile):
    with open(outfile, 'wb') as f:
        pickle.dump(data, f)

def generate_data(env, agent, outfile, num_episodes=100):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []


    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        states.append(obs)

        while not done:
            action = agent.action(obs)
            actions.append(action)
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)
            dones.append(done)
            if not done:
                states.append(obs)
                next_states.append(obs)

    data = {
        "states" : np.array(states),
        "actions" : np.array(actions),
        "rewards" : np.array(rewards),
        "next_states" : np.array(next_states),
        "dones" : np.array(done)
    }

    save(data, outfile)

    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", "-n", default=100, type=int)
    parser.add_argument("--outfile_name", "-o", default="out.pkl", type=str)
    parser.add_argument("--outfile_dir", "-d", default=CURR_DIR, type=str)

    env = MyEnv(size=9, good_goal_pos=(7, 8), bad_goal_pos=(8, 7))
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)

    args = parser.parse_args()

    agent = GoToGoodGoalAgent(observation_space=env.observation_space, action_space=env.action_space)

    generate_data(env, agent, os.path.join(args.outfile_dir, args.outfile_name), args.num_episodes)

if __name__ == '__main__':
    main()