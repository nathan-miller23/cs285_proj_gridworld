import os, sys, argparse
import numpy as np

from gym_minigrid.envs.mygridworld import MyEnv
from gym_minigrid.wrappers import *
from agents.hardcoded_agents import GoToGoodGoalAgent
from utils import save, load

CURR_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CURR_DIR, 'data')

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
            next_states.append(obs)
            timesteps_collected += 1
            if not done:
                states.append(obs)
                

    data = {
        "states" : np.array(states),
        "actions" : np.array(actions),
        "rewards" : np.array(rewards),
        "next_states" : np.array(next_states),
        "dones" : np.array(dones)
    }

    save(data, outfile)

    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_timesteps", "-n", default=10000, type=int)
    parser.add_argument("--outfile_name", "-o", default="out", type=str)
    parser.add_argument("--outfile_dir", "-d", default=DATA_DIR, type=str)
    parser.add_argument('--save_agent', '-sa', action='store_true')
    parser.add_argument('--save_environment', '-se', action='store_true')
    parser.add_argument('--epsilon', '-eps', default=0.0, type=float)
    parser.add_argument('--delta', '-dlt', default=0.0, type=float)
    parser.add_argument('--gamma', '-gm', default=1.0, type=float)
    args = parser.parse_args()

    env = MyEnv(size=9, good_goal_pos=(7, 8), bad_goal_pos=(8, 7), gamma=args.gamma)
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)

    agent = GoToGoodGoalAgent(epsilon=args.epsilon, delta=args.delta,
                              observation_space=env.observation_space, action_space=env.action_space)

    if args.save_agent:
        agent_save_loc = os.path.join(args.outfile_dir, args.outfile_name, "agent.pkl")
        save(agent, agent_save_loc)
    if args.save_environment:
        env_save_loc = os.path.join(args.outfile_dir, args.outfile_name, "env.pkl")
        save(env, env_save_loc)
    
    data_save_loc = os.path.join(args.outfile_dir, args.outfile_name, "data.pkl")
    generate_data(env, agent, data_save_loc, args.num_timesteps)

if __name__ == '__main__':
    main()