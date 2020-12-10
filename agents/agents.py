import numpy as np

class Agent():

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def action(self, observation):
        probs = self.action_probs(observation)
        return np.random.choice(len(probs), p=probs)

    def actions(self, observations):
        actions = []
        for obs in observations:
            actions.append(self.action(obs))

        return np.array(actions)

    def action_probs(self, observation):
        raise NotImplementedError()