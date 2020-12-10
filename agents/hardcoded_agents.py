import numpy as np

from gym_minigrid.minigrid import IDX_TO_OBJECT, IDX_TO_COLOR, MyMiniGridEnv
from utils import mannhattan_distance
from agents.agents import Agent


class RandAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def action_probs(self, observation):
        n = self.action_space.n
        unif_prob = 1 / n
        return np.ones(n) * unif_prob

class NoisyGreedyAgent(Agent):
    def __init__(self, delta=0.0, epsilon=0.0, **kwargs):
        super().__init__(**kwargs)
        if not (delta >= 0 and delta <= 1):
            raise ValueError("Expected {} in range [0, 1] but got {}".format("delta", delta))
        if not (epsilon >= 0 and epsilon <= 1):
            raise ValueError("Expected {} in range [0, 1] but got {}".format("epsilon", epsilon))
        self.delta = delta
        self.epsilon = epsilon

    def _get_greedy_actions(self, observation):
        raise NotImplementedError("Need to define greedy actions")

    def _get_positions(self, observation):
        bad_goal_pos = None
        good_goal_pos = None
        curr_pos = None

        for i in range(len(observation)):
            for j in range(observation.shape[1]):
                obj_state = observation[i, j, :]
                if IDX_TO_OBJECT[obj_state[0]] == 'goal' and IDX_TO_COLOR[obj_state[1]] == 'green':
                    good_goal_pos = (i, j)
                elif IDX_TO_OBJECT[obj_state[0]] == 'goal' and IDX_TO_COLOR[obj_state[1]] == 'red':
                    bad_goal_pos = (i, j)
                elif IDX_TO_OBJECT[obj_state[0]] == 'agent':
                    curr_pos = (i, j)
        
        if not good_goal_pos or not curr_pos or not bad_goal_pos:
            raise ValueError("Expected to find positions but instead got\nGood Goal: {}\nBad Goal: {}\nAgent: {}".format(good_goal_pos, bad_goal_pos, curr_pos))

        return good_goal_pos, bad_goal_pos, curr_pos

    def action_probs(self, observation):
        probs = [0] * self.action_space.n
        proposed_actions = self._get_greedy_actions(observation)
        other_actions = list(set(MyMiniGridEnv.Actions) - set([MyMiniGridEnv.Actions.stay]))

        probs[MyMiniGridEnv.Actions.stay.value] = self.delta
        for action in other_actions:
            probs[action.value] += (1/len(other_actions)) * (1 - self.delta) * self.epsilon
        for action in proposed_actions:
            probs[action.value] += (1/len(proposed_actions)) * (1 - self.delta) * (1 - self.epsilon)

        return probs

class RandGoodGoalAgent(NoisyGreedyAgent):
    def __init__(self, expected_good_goal_pos=(7, 8), **kwargs):
        super().__init__(**kwargs)
        self.expected_good_goal_pos = expected_good_goal_pos

    def _get_greedy_actions(self, observation):
        good_goal_pos, _, curr_pos = self._get_positions(observation)

        if not good_goal_pos == self.expected_good_goal_pos:
            raise ValueError("This hardcoded agent expects good_goal_pos to be {} but got {}".format(self.expected_good_goal_pos, good_goal_pos))

        if mannhattan_distance(curr_pos, good_goal_pos) == 1:
            proposed_actions = [MyMiniGridEnv.Actions.down]
        else:
            proposed_actions = list(set(MyMiniGridEnv.Actions) - set([MyMiniGridEnv.Actions.stay]))
        
        return proposed_actions



class GoToGoodGoalAgent(NoisyGreedyAgent):
    def __init__(self, expected_good_goal_pos=(7, 8), **kwargs):
        super().__init__(**kwargs)
        self.expected_good_goal_pos = expected_good_goal_pos

    def _get_greedy_actions(self, observation):
        good_goal_pos, _, curr_pos = self._get_positions(observation)

        if not good_goal_pos == self.expected_good_goal_pos:
            raise ValueError("This hardcoded agent expects good_goal_pos to be {} but got {}".format(self.expected_good_goal_pos, good_goal_pos))

        agent_x, agent_y = curr_pos

        if mannhattan_distance(curr_pos, good_goal_pos) == 1:
            proposed_actions = [MyMiniGridEnv.Actions.down]
        elif agent_x == agent_y:
            proposed_actions = [MyMiniGridEnv.Actions.right, MyMiniGridEnv.Actions.down]
        elif agent_x > agent_y:
            proposed_actions = [MyMiniGridEnv.Actions.down]
        else:
            proposed_actions = [MyMiniGridEnv.Actions.right]

        return proposed_actions

        
