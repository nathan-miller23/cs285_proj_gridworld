import os, sys, argparse
import numpy as np

from gym_minigrid.envs.mygridworld import MyEnv
from gym_minigrid.wrappers import *
from agents.hardcoded_agents import GoToGoodGoalAgent
from utils import save, load, DATA_DIR

def generate_data(env, agent, outfile, num_timesteps, collect_rbg, tile_size):
    states_rbg = []
    states = []
    actions = []
    rewards = []
    next_states = []
    next_states_rbg = []
    dones = []

    timesteps_collected = 0


    while timesteps_collected < num_timesteps:
        obs = env.reset()
        states.append(obs)
        if collect_rbg:
            obs_rbg = env.render(mode='rbg_array', highlight=False, tile_size=tile_size)
            states_rbg.append(obs_rbg)
        done = False
        
        while not done:
            action = agent.action(obs)
            obs, reward, done, _ = env.step(action)
            obs_rbg = env.render(mode='rbg_array', highlight=False, tile_size=tile_size)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_states.append(obs)

            if collect_rbg:
                next_states_rbg.append(obs_rbg)
            timesteps_collected += 1
            if not done:
                states.append(obs)
                if collect_rbg:
                    states_rbg.append(obs_rbg)

                

    data = {
        "states" : np.array(states),
        "actions" : np.array(actions),
        "rewards" : np.array(rewards),
        "next_states" : np.array(next_states),
        "dones" : np.array(dones)
    }
    
    if collect_rbg:
        data['states_rbg'] = np.array(states_rbg)
        data['next_states_rbg'] = np.array(next_states_rbg)

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
    parser.add_argument('--collect_rbg', '-rbg', action='store_true')
    parser.add_argument('--tile_size', '-t', default=8, type=int)
    args = parser.parse_args()

    base_env = MyEnv(size=9, good_goal_pos=(7, 8), bad_goal_pos=(8, 7), gamma=args.gamma)
    env = FullyObsWrapper(base_env)
    env = ImgObsWrapper(env)

    agent = GoToGoodGoalAgent(epsilon=args.epsilon, delta=args.delta,
                              observation_space=env.observation_space, action_space=env.action_space)

    if args.save_agent:
        agent_save_loc = os.path.join(args.outfile_dir, args.outfile_name, "agent.pkl")
        save(agent, agent_save_loc)
    if args.save_environment:
        env_save_loc = os.path.join(args.outfile_dir, args.outfile_name, "env.pkl")
        save(env, env_save_loc)

        if args.collect_rbg:
            rbg_env_save_loc = os.path.join(args.outfile_dir, args.outfile_name, "rbg_env.pkl")
            rbg_env = RGBImgObsWrapper(base_env, tile_size=args.tile_size)
            rbg_env = ImgObsWrapper(rbg_env)
            save(rbg_env, rbg_env_save_loc)
    
    data_save_loc = os.path.join(args.outfile_dir, args.outfile_name, "data.pkl")
    generate_data(env, agent, data_save_loc, args.num_timesteps, args.collect_rbg, args.tile_size)

if __name__ == '__main__':
    main()