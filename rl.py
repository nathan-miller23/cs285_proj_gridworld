import tqdm

from agents.hardcoded_agents import GoToGoodGoalAgent
from gym_minigrid.envs.mygridworld import MyEnv
from gym_minigrid.minigrid import IDX_TO_OBJECT
from gym_minigrid.wrappers import *
from utils import mannhattan_distance


def tabular_learning(env, agent, gamma, max_iters=1e3, max_tol=1e-4, state_func=False):
    """Runs tabular RL on the enironment to compute the state functions exactly

    Args:
        env (gym_minigrid.MyEnv): Environment on which to perform tabular RL
        agent (agents.Agent): Defines the policy that the value/Q functions will be computed wrt
        gamma (float): Discount factor
        max_iters (int, optional): Maximum number of value iterations to perform before convergence. Defaults to 1e3.
        max_tol (int, optional): Relative tolerance (between successive value iters) that must be met for convergence. Defaults to 1e-4.
        state_func (bool, optional): Whether returned functions should accept states (if True) or observations (if False). 
            A 'state' is the minimal encoding (tuple of agent x,y position). An observation is the full spatially encoded 
            numpy array. Defaults to False.

    Returns:
        (value_function, q_function, A_strat_function): Tuple of state functions of interest

        Note: The input of the returned functions is either an encoded 'state' (agent position) or full observation, 
        depending on the value passed for `state_func` parameter
    """
    values_table = _value_iteration(env, agent, gamma, max_iters, max_tol)

    value_function = _get_value_function(values_table, state_func)
    q_function = _get_q_function(values_table, env, gamma, state_func)
    A_strat_function = _get_a_strat_function(values_table, env, gamma, state_func)

    return value_function, q_function, A_strat_function


def _value_iteration(env, agent, gamma=0.9, max_iters=1e4, max_tol=1e-4):
    values_t = get_initial_value_table(env)

    t = tqdm.trange(int(max_iters))
    for i in t:
        delta = 0.0
        values_t_1 = values_t.copy()

        q_func = _get_q_function(values_t, env, gamma, state_func=False)

        for state in values_t.keys():
            if (hasattr(env, 'good_goal_pos') and state == env.good_goal_pos) or (hasattr(env, 'bad_goal_pos') and state == env.bad_goal_pos):
                continue
            obs = decode(state, env)
            action_probs = agent.action_probs(obs)
            value = 0
            for action, prob in enumerate(action_probs):
                q_val = q_func(obs, action)
                value += prob * q_val

            values_t_1[state] = value
            delta = max(delta, abs(values_t_1[state] - values_t[state]))

        t.set_description("Delta Value for Value-Learning {:.4f}".format(delta))
        converged = delta < max_tol
        values_t = values_t_1
        if converged:
            break

    return values_t


def _get_value_function(values_table, state_func):
    def value_function_state(state):
        return values_table[state]

    def value_function_obs(obs):
        return value_function_state(encode(obs))

    return value_function_state if state_func else value_function_obs


def _get_q_function(values_table, env, gamma, state_func):
    def q_function_state(state, action):
        env.reset()
        env.unwrapped.agent_pos = state
        obs_prime, reward, _, _ = env.step(action)
        state_prime = encode(obs_prime)
        return reward + gamma * values_table[state_prime]

    def q_function_obs(obs, action):
        return q_function_state(encode(obs), action)

    return q_function_state if state_func else q_function_obs


def _get_a_strat_function(values_table, env, gamma, state_func):
    q_function_state = _get_q_function(values_table, env, gamma, True)

    def A_strat_state(s):
        return max([q_function_state(s, a) for a in env.Actions]) - min([q_function_state(s, a) for a in env.Actions])

    def A_strat_obs(obs):
        return A_strat_state(encode(obs))

    return A_strat_state if state_func else A_strat_obs


def encode(obs):
    """Converts full gridworld observation into minimal state encoding (agent position)

    Args:
        obs (np.array): Full spatially encoded array of the gridworld state

    Returns:
        (Tuple(int, int)): Agent (x, y) coordinate position
    """
    return _find_agent(obs)[0]


def decode(state, env):
    """Converts minimal state encoding (agent position) to higher dimension obseration

    Args:
        state (Tuple(int, int)): Agent (x, y) coordinate position
        env (MyEnv): Environment instance from whence the state came. NOTE: env is mutated

    Returns:
        (np.array): Full spatially encoded array of the gridworld state
    """
    env.reset()
    env.unwrapped.agent_pos = state
    obs = env.step(env.Actions.stay)[0]
    return obs


def get_initial_value_table(env):
    values = {}
    all_states = env.get_accessible_states()
    for state in all_states:
        values[state] = 0
    return values


def get_value_table_from_obs(env, obs_func):
    state_func = lambda state: obs_func(decode(state, env))
    return get_value_table_from_states(env, state_func)


def get_value_table_from_states(env, state_func):
    states = env.get_empty_states()
    mat = np.zeros((env.width, env.height))
    for state in states:
        val = state_func(state)
        mat[state[0], state[1]] = val
    return mat


def _find_agent(state):
    for i in range(len(state)):
        for j in range(state.shape[1]):
            encoding = state[i, j, :]
            if IDX_TO_OBJECT[encoding[0]] == 'agent':
                return (i, j), encoding.copy()

    raise ValueError("Agent not found!")


if __name__ == '__main__':
    # TODO: make this an argparse script
    env = MyEnv(size=9, good_goal_pos=(7, 8), bad_goal_pos=(8, 7))
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    agent = GoToGoodGoalAgent(observation_space=env.observation_space, action_space=env.action_space, epsilon=0.0)
    gamma = 0.9

    v_pi, q_pi, a_strat_pi = tabular_learning(env, agent, gamma=gamma)

    # Sanity check test to ensure value function properly computed
    states = env.get_empty_states()
    for state in states:
        expected = gamma ** (mannhattan_distance(state, env.good_goal_pos) - 1) * 10
        actual = v_pi(decode(state, env))
        if abs(expected - actual) > 1e-4:
            raise ValueError("Expected: {} got: {} for state {}".format(expected, actual, state))

## TODO
# Write 'set_state' for gridworld value iteration to make this cleaner
