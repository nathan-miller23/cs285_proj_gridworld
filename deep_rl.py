from utils import ReplayBuffer
from torch import nn, optim
from agents.network import Net
import numpy as np
import torch, tqdm


class DoubleQNet:
    def __init__(self, state_dim, n_actions, gamma=0.99, lmbda=1.0, eps=1e-3, itr_target_update=1e1, device="cuda"): # TODO add ability to customize architecture
        self.q_net = Net(state_dim, n_actions).to(device)
        self.q_net_opt = optim.Adam(self.q_net.parameters(), lr=0.001)
        self.target_q_net = Net(state_dim, n_actions).to(device)

        self.itr_target_update = itr_target_update
        self.lmbda = lmbda
        self.count = 0
        self.gamma = gamma
        self.eps = eps
        self.device = device
        self.loss_func = nn.MSELoss()
        self.memory = ReplayBuffer(1e4, 64)

    def forward(self, x):
        return self.q_net(x.permute(0, 3, 1, 2))

    def add_data(self, s, a, r, s_, a_, d):
        self.memory.add(s, a, r, s_, a_, d)

    def a_strat(self, s): 
        s = torch.as_tensor(s[np.newaxis, :], dtype=torch.float32).to(self.device).permute(0, 3, 1, 2)
        q_max = torch.max(self.q_net(s).gather(1, torch.argmax(self.target_q_net(s), dim=1).unsqueeze(1)))
        q_min = torch.max(self.q_net(s).gather(1, torch.argmin(self.target_q_net(s), dim=1).unsqueeze(1)))
        q_rnd = torch.max(torch.abs(self.q_net(s) - self.target_q_net(s)))
        return (torch.clamp(q_max - q_min, min=self.eps) + self.lmbda * q_rnd).detach().cpu().item()

    def train_step(self, b_states, b_actions, b_rewards, b_next_states, b_next_actions, b_done_masks):
        b_states = torch.as_tensor(b_states, dtype=torch.float32).to(self.device).permute(0, 3, 1, 2)
        b_actions = torch.as_tensor(b_actions, dtype=torch.int64).to(self.device)
        b_rewards = torch.as_tensor(b_rewards[:, np.newaxis], dtype=torch.float32).to(self.device)
        b_next_states = torch.as_tensor(b_next_states, dtype=torch.float32).to(self.device).permute(0, 3, 1, 2)
        b_done_masks = torch.as_tensor(b_done_masks[:, np.newaxis], dtype=torch.float32).to(self.device)
        b_next_actions = torch.as_tensor(b_next_actions, dtype=torch.int64).to(self.device)

        target_q_val = b_rewards + (1.0 - b_done_masks) * self.gamma * self.target_q_net(b_next_states).gather(1, b_next_actions.unsqueeze(1))
        loss = self.loss_func(self.q_net(b_states).gather(1, b_actions.unsqueeze(1)), target_q_val.detach())

        self.q_net_opt.zero_grad()
        loss.backward()
        self.q_net_opt.step()

        self.count += 1
        if self.count >= self.itr_target_update:
            self.count = 0
            for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
                target_param.data.copy_(param.data)
        return loss.detach().cpu().numpy()

    def train(self, num_iterations):
        losses = []
        running_loss = 1
        t = tqdm.trange(int(num_iterations))
        for _ in t:
            loss = self.train_step(*self.memory.sample())
            running_loss = running_loss * 0.99  + 0.01 * loss
            t.set_description("Q-Learning Loss {:.4f}".format(running_loss))
            losses.append(loss)
        return losses
   

def train_q_network(env, expert_data_dict, max_iters=1e4, gamma=0.9, lmbda=1.0, itr_target_update=1, use_cuda=False, dataset_size=1000):
    ddq_net = DoubleQNet(env.observation_space.shape, env.action_space.n, gamma=gamma, lmbda=lmbda, itr_target_update=itr_target_update, device="cuda" if use_cuda else "cpu")
    states = expert_data_dict["states"][:dataset_size]
    actions = expert_data_dict["actions"][:dataset_size]
    rewards = expert_data_dict["rewards"][:dataset_size]
    next_states = expert_data_dict["next_states"][:dataset_size]
    next_actions = actions[1:dataset_size + 1]
    dones = expert_data_dict["dones"][:dataset_size]
    
    print("Adding data")
    for s, a, r, s_, a_, d in zip(states, actions, rewards, next_states, next_actions, dones):
        ddq_net.add_data(s, a, r, s_, a_, d)
    print("Finished adding data")
    ddq_net.train(max_iters)

    return ddq_net.a_strat