import numpy as np

from gym_minigrid.minigrid import IDX_TO_OBJECT, IDX_TO_COLOR, MyMiniGridEnv


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
    def __init__(self, delta=0.0, epsilon=0.0, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.delta = delta

    def action(self, observation):
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

        agent_x, agent_y = curr_pos
        goal_x, goal_y = good_goal_pos

        if agent_x > goal_x:
            proposed_action = MyMiniGridEnv.Actions.left
        elif agent_x < goal_x:
            proposed_action = MyMiniGridEnv.Actions.right
        elif agent_y > goal_y:
            proposed_action = MyMiniGridEnv.Actions.up
        elif agent_y < goal_y:
            proposed_action = MyMiniGridEnv.Actions.down
        else:
            proposed_action = MyMiniGridEnv.Actions.stay

        if np.random.random() < self.delta:
            return MyMiniGridEnv.Actions.stay
        if np.random.random() < self.epsilon:
            return np.random.choice([MyMiniGridEnv.Actions.left, MyMiniGridEnv.Actions.right, MyMiniGridEnv.Actions.up,
                                     MyMiniGridEnv.Actions.down, MyMiniGridEnv.Actions.stay], size=1, p=[0.2] * 5)[0]
        return proposed_action

    def action_probs(self, observation):
        probs = np.zeros(self.action_space.n)
        action = self.action(observation)
        probs[action] = 1
        return probs
