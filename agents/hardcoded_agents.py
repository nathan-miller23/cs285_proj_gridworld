import numpy as np

from gym_minigrid.minigrid import IDX_TO_OBJECT, IDX_TO_COLOR, MyMiniGridEnv
from utils import mannhattan_distance


class Agent():

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def action(self, observation):
        raise NotImplementedError()

    def actions(self, observations):
        actions = []
        for obs in observations:
            actions.append(self.action(obs))

        return np.array(actions)

    def action_probs(self, observation):
        raise NotImplementedError()


class RandAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def action(self, observation):
        return self.action_space.sample()

    def action_probs(self, observation):
        unif_prob = 1 / self.action_space.n
        return np.ones() * unif_prob


class GoToGoodGoalAgent(Agent):
    def __init__(self, delta=0.0, epsilon=0.0, expected_good_goal_pos=(7, 8), **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.expected_good_goal_pos = expected_good_goal_pos

    def action(self, observation):
        probs = self.action_probs(observation)
        return np.random.choice(len(probs), p=probs)

    def action_probs(self, observation):
        probs = [None] * self.action_space.n
        good_goal_pos = None
        curr_pos = None

        for i in range(len(observation)):
            for j in range(observation.shape[1]):
                obj_state = observation[i, j, :]
                if IDX_TO_OBJECT[obj_state[0]] == 'goal' and IDX_TO_COLOR[obj_state[1]] == 'green':
                    good_goal_pos = (i, j)
                elif IDX_TO_OBJECT[obj_state[0]] == 'agent':
                    curr_pos = (i, j)
        
        if not good_goal_pos or not curr_pos:
            print(good_goal_pos)
            print(curr_pos)
            raise ValueError()

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

        other_actions = list(set(MyMiniGridEnv.Actions) - set(proposed_actions + [MyMiniGridEnv.Actions.stay]))

        probs[MyMiniGridEnv.Actions.stay.value] = self.delta
        for action in other_actions:
            probs[action.value] = (1/len(other_actions)) * (1 - self.delta) * self.epsilon
        for action in proposed_actions:
            probs[action.value] = (1/len(proposed_actions)) * (1 - self.delta) * (1 - self.epsilon)

        return probs

        
