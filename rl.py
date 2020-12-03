from gym_minigrid.minigrid import IDX_TO_OBJECT
from gym_minigrid.envs.mygridworld import MyEnv
from gym_minigrid.wrappers import *
from agents.hardcoded_agents import GoToGoodGoalAgent
import numpy as np


def tabular_learning(env, agent, gamma, max_iters=1e4, max_tol=1e-4):
    values_table = _value_iteration(env, agent, gamma, max_iters, max_tol)

    value_function = _get_value_function(values_table)
    q_function = _get_q_function(values_table, env)

    return value_function, q_function


def _value_iteration(env, agent, gamma=0.9, max_iters=1e4, max_tol=1e-4):
    values_t = get_initial_value_table(env)
    converged = False
    i = 0

    while not converged:
        delta = 0.0
        values_t_1 = values_t.copy()

        q_func = _get_q_function(values_t, env)
        
        for state in values_t.keys():
            if state == env.good_goal_pos or state == env.bad_goal_pos:
                continue
            obs = decode(state, env)
            action_probs = agent.action_probs(obs)
            value = 0
            for action, prob in enumerate(action_probs):
                q_val = q_func(obs, action)
                value += prob * q_val
                 
            values_t_1[state] = value
            delta = max(delta, abs(values_t_1[state] - values_t[state]))

        converged = i > max_iters or delta < max_tol
        values_t = values_t_1
        i += 1

    return values_t

def _get_value_function(values_table):
    def value_function(obs):
        return values_table[encode(obs)]
    return value_function

def _get_q_function(values_table, env):
    def q_function(obs, action):
        state = encode(obs)
        env.reset()
        env.env.env.agent_pos = state
        obs_prime, reward, _, _ = env.step(action)
        state_prime = encode(obs_prime)
        return reward + gamma * values_table[state_prime]
    return q_function


def encode(state):
    return _find_agent(state)[0]

def decode(state, env):
    env.reset()
    env.env.env.agent_pos = state
    obs = env.step(env.Actions.stay)[0]
    return obs


def get_initial_value_table(env):
    values = {}
    for i in range(1, env.width-1):
        for j in range(1, env.height-1):
            values[(i, j)] = 0
    values[env.good_goal_pos] = values[env.bad_goal_pos] = 0
    return values

def _find_agent(state):
    for i in range(len(state)):
        for j in range(state.shape[1]):
            encoding = state[i, j, :]
            if IDX_TO_OBJECT[encoding[0]] == 'agent':
                return (i, j), encoding

    raise ValueError("Agent not found!")

def _mannhattan_distance(state_1, state_2):
    return abs(state_1[0] - state_2[0]) + abs(state_1[1] - state_2[1])


if __name__ == '__main__':
    env = MyEnv(size=9, good_goal_pos=(7, 8), bad_goal_pos=(8, 7))
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    agent = GoToGoodGoalAgent(observation_space=env.observation_space, action_space=env.action_space)

    gamma = 0.9

    v_pi, q_pi = tabular_learning(env, agent, gamma=gamma)

    states = list(get_initial_value_table(env).keys())
    states.remove(env.good_goal_pos)
    states.remove(env.bad_goal_pos)

    for state in states:
        expected = gamma**(_mannhattan_distance(state, env.good_goal_pos) - 1) * 10
        actual = v_pi(decode(state, env))
        if abs(expected - actual) > 1e-4:
            raise ValueError("Expected: {} got: {} for state {}".format(expected, actual, state))



## TODO
# Write 'set_state' for gridworld value iteration to make this cleaner