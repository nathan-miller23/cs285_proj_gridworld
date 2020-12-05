import os, sys, argparse, pickle
import numpy as np

from gym_minigrid.envs.mygridworld import MyEnv
from gym_minigrid.wrappers import *
from agents.hardcoded_agents import GoToGoodGoalAgent

CURR_DIR = os.path.abspath('.')

def save(data, outfile):
    with open(outfile, 'wb') as f:
        pickle.dump(data, f)

def load(outfile):
    with open(outfile, 'rb') as f:
        return pickle.load(f)

def generate_data(env, agent, outfile, num_timesteps=100):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    timesteps_collected = 0


    while timesteps_collected < num_timesteps:
        timesteps_collected += 1
        obs = env.reset()
        done = False
        states.append(obs)

        while not done:
            action = agent.action(obs)
            actions.append(action)
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)
            dones.append(done)
            timesteps_collected += 1
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
    parser.add_argument("--num_timesteps", "-n", default=10000, type=int)
    parser.add_argument("--outfile_name", "-o", default="out", type=str)
    parser.add_argument("--outfile_dir", "-d", default=CURR_DIR, type=str)
    parser.add_argument('--save_agent', '-sa', action='store_true')
    parser.add_argument('--save_environment', '-se', action='store_true')
    args = parser.parse_args()

    env = MyEnv(size=9, good_goal_pos=(7, 8), bad_goal_pos=(8, 7))
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)

    agent = GoToGoodGoalAgent(observation_space=env.observation_space, action_space=env.action_space)

    if args.save_agent:
        agent_save_loc = os.path.join(args.outfile_dir, "{}_agent.pkl".format(args.outfile_name))
        save(agent, agent_save_loc)
    if args.save_environment:
        env_save_loc = os.path.join(args.outfile_dir, "{}_env.pkl".format(args.outfile_name))
        save(env, env_save_loc)

    generate_data(env, agent, os.path.join(args.outfile_dir, "{}_data.pkl".format(args.outfile_name)), args.num_timesteps)

if __name__ == '__main__':
    main()