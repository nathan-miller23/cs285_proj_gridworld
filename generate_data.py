import os, sys, argparse
import numpy as np

from gym_minigrid.envs.mygridworld import MyEnv
from gym_minigrid.envs.tightrope import TightRopeEnv
from gym_minigrid.wrappers import *
from agents.hardcoded_agents import GoToGoodGoalAgent, TighRopeExpert
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
    """Simulate an Agent in an environment and store trajectory data transitions + other metadata

    Arguments:
        num_timesteps (int): Number of (s,a,r,s') transitions to collect. Note that, because we collect at the trajectory level
            not the timestep level, the actual number of timesteps collected could be slightly higher
        outfile_name (str): sud-directory name for this particular data collection
        outfile_dir (str): Root directory in which data will be stored
        save_agent (bool): Whether to save a pickled copy of the agent that generated the data (could be used for on-policy RL)
        save_environment (bool): Whether to save a pickled copy of the environment in which the data was generated (could be used for tabular RL)
        epsilon (float): What percentage agent peforms random action (only used if tightrope_env=False)
        standard_epsilon (float): Percentage of time agent performs random action when not on tightrope (only used if tightrope_env=True)
        critical_epsilon (float): Percentage fo time agent performs random action when on tightrope (only used if tightrope_env=True)
        delta (float): Percentage of time agent performs "STAY" action
        gamma (float): Discount factor of environment
        collect_rbg (bool): Whether we should collect RBG observations in addition to our standard gridworld observations
        tile_size (int): Number of pixels per gridworld "space". Only relevant if collect_rbg=True
        tightrope_env (bool): Whether to use TightRopeEnv (if True). Otherwise use standard good/bad goal environment

    After running, the following directory structure should exist

    /outfile_dir/
        outfile_name/
            data.pkl
            agent.pkl (if save_agent=True)
            env.pkl (if save_env=True)
            rbg_env.pkl (if collect_rbg=True)

    data.pickle is a dictionary of the following form:
        "states" : np.array() shape (N, *obs_shape)
        "actions" np.array shape (N, len(action_space))
        "rewards" np.array shape (N,)
        "next_states" np.array shape (N, *obs_shape)
        "dones" np.array shape (N,)

        "states_rbg" np.array shape (N, H, W, 3) (if collect_rbg=True, otherwise key doesn't exist)
        "next_states_rbg" np.array shape (N, H, W, 3) (if collect_rbg=True, otherwise key doesn't exist)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_timesteps", "-n", default=10000, type=int)
    parser.add_argument("--outfile_name", "-o", default="out", type=str)
    parser.add_argument("--outfile_dir", "-d", default=DATA_DIR, type=str)
    parser.add_argument('--save_agent', '-sa', action='store_true')
    parser.add_argument('--save_environment', '-se', action='store_true')
    parser.add_argument('--epsilon', '-eps', default=0.0, type=float)
    parser.add_argument('--standard_epsilon', '-stdeps', default=0.0, type=float)
    parser.add_argument('--critical_epsilon', '-criteps', default=0.0, type=float)
    parser.add_argument('--delta', '-dlt', default=0.0, type=float)
    parser.add_argument('--gamma', '-gm', default=1.0, type=float)
    parser.add_argument('--collect_rbg', '-rbg', action='store_true')
    parser.add_argument('--tile_size', '-t', default=8, type=int)
    parser.add_argument('--tightrope_env', '-tight', action='store_true')
    args = parser.parse_args()

    base_env = TightRopeEnv(gamma=args.gamma) if args.tightrope_env else  MyEnv(good_goal_pos=(7, 8), bad_goal_pos=(8, 7), gamma=args.gamma)
    env = FullyObsWrapper(base_env)
    env = ImgObsWrapper(env)

    agent = None
    if args.tightrope_env:
        agent = TighRopeExpert(delta=args.delta, 
                               observation_space=env.observation_space, 
                               action_space=env.action_space, 
                               standard_epsilon=args.standard_epsilon, 
                               critical_epsilon=args.critical_epsilon)
    else: 
        agent = GoToGoodGoalAgent(epsilon=args.epsilon, 
                                  delta=args.delta, 
                                  observation_space=env.observation_space, 
                                  action_space=env.action_space)

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