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

    def _get_positions(self, observation, expect_all=True):
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
        
        if not curr_pos:
            raise ValueError("Expected to find agent but didn't!")
        if expect_all and not (good_goal_pos and bad_goal_pos):
            raise ValueError("Expected to find positions but instead got\nGood Goal: {}\nBad Goal: {}".format(good_goal_pos, bad_goal_pos))

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

        
class TighRopeExpert(NoisyGreedyAgent):
    def __init__(self, standard_epsilon=0.5, critical_epsilon=0.1, **kwargs):
        super().__init__(**kwargs)
        self.standard_epsilon = standard_epsilon
        self.critical_epsilon = critical_epsilon

    def _get_adjacent(self, observation):
        _, _, curr_pos = self._get_positions(observation, expect_all=False)

        obj_up_state = observation[curr_pos[0], curr_pos[1] - 1, :]
        obj_down_state = observation[curr_pos[0], curr_pos[1] + 1, :]
        obj_left_state = observation[curr_pos[0] - 1, curr_pos[1], :]
        obj_right_state = observation[curr_pos[0] + 1, curr_pos[1], :]

        return [obj_up_state, obj_down_state, obj_left_state, obj_right_state]

    def _get_is_empty_adjecent(self, observation):
        return [IDX_TO_OBJECT[obj_state[0]] == 'empty' for obj_state in self._get_adjacent(observation)]

    def _get_is_lava_adjacent(self, observation):
        return [IDX_TO_OBJECT[obj_state[0]] == 'goal' and IDX_TO_COLOR[obj_state[1]] == 'red' for obj_state in self._get_adjacent(observation)]

    def _is_touching_lava(self, observation):
        return any(self._get_is_lava_adjacent(observation))

    def _get_greedy_actions(self, observation):
        good_goal_pos, _, curr_pos = self._get_positions(observation, expect_all=False)
        _, _, is_lava_left, is_lava_right = self._get_is_lava_adjacent(observation)
        _, is_empty_down, is_empty_left, is_empty_right = self._get_is_empty_adjecent(observation)

        mid_x = self.observation_space.shape[0] // 2
        mid_y = self.observation_space.shape[1] // 2

        agent_x, agent_y = curr_pos

        # If on tightrope, above goal, or in horizontal center, go down
        if is_lava_left and is_lava_right or mannhattan_distance(curr_pos, good_goal_pos) == 1 or agent_x == mid_x: 
            proposed_actions = [MyMiniGridEnv.Actions.down]
        # If in bottom half or top left, go down and right (if not blocked)
        elif agent_y > mid_y or agent_x < mid_x:
            proposed_actions = []
            if is_empty_down:
                proposed_actions.append(MyMiniGridEnv.Actions.down)
            if is_empty_right:
                proposed_actions.append(MyMiniGridEnv.Actions.right)
        # if in top right, go down and left (if not blocked)
        else:
            proposed_actions = []
            if is_empty_down:
                proposed_actions.append(MyMiniGridEnv.Actions.down)
            if is_empty_left:
                proposed_actions.append(MyMiniGridEnv.Actions.left)


        return proposed_actions
    

    def action_probs(self, observation):
        if self._is_touching_lava(observation):
            self.epsilon = self.critical_epsilon
        else:
            self.epsilon = self.standard_epsilon
        return super().action_probs(observation)