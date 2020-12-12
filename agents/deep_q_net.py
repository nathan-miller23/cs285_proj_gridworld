import numpy as np
import torch
from torch import nn, optim

from agents.replay_buffer import ReplayBuffer


class QNet(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden_layers=2, hidden_dim=64):
        super(QNet, self).__init__()
        self.model = self._build_model(n_states, n_actions, n_hidden_layers, hidden_dim)

    @staticmethod
    def _build_model(n_states, n_actions, n_hidden_layers, hidden_dim):
        layers = [nn.Linear(n_states, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, n_actions))
        return nn.Sequential(*layers)

    def forward(self, s):
        return self.model(s)

def encode(state):
    return _find_agent(state)[0]


def decode(state, env):
    env.reset()
    env.unwrapped.agent_pos = state
    obs = env.step(env.Actions.stay)[0]
    return obs


class DoubleQNet:
    def __init__(self, n_states, n_actions, n_hidden_layers=2, hidden_dim=64, gamma=0.99, itr_target_update=1e1,
                 device="cuda", state_func=False):
        self.q_net = QNet(n_states, n_actions, n_hidden_layers, hidden_dim)
        self.q_net_opt = optim.Adam(self.q_net.parameters(), lr=0.001)
        self.target_q_net = QNet(n_states, n_actions, n_hidden_layers, hidden_dim)

        self.itr_target_update = itr_target_update
        self.count = 0
        self.gamma = gamma
        self.device = device
        self.memory = ReplayBuffer(1e4, 64)

    def forward(self, x):
        return self.q_net(x)

    def advantage(self, s, r):
        pass

    def q(self, x, a):
        if (not state_func):
            x = encode(x)

    def add_data(self, s, a, r, s_, d):
        self.memory.add(s, a, r, s_, d)

    def train_step(self, b_states, b_actions, b_rewards, b_next_states, b_done_masks):
        target_actions = torch.argmax(self.target_q_net(b_next_states), dim=1)
        target_q_val = b_rewards + b_done_masks * self.gamma * self.q_net(b_next_states).\
            gather(1, target_actions.unsqueeze(1))
        loss = nn.MSELoss(self.q_net(b_states).gather(1, b_actions.unsqueeze(1)), target_q_val.detach())

        self.q_net_opt.zero_grad()
        loss.backwards()
        self.q_net_opt.step()

        self.count += 1
        if self.count >= self.itr_target_update:
            self.count = 0
            for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
                target_param.data.copy_(param.data)
        return loss.detach().cpu().numpy()

    def train(self, num_iterations):
        losses = []
        for _ in num_iterations:
            losses.append(self.train_step(*self.memory.sample()))
        return losses
