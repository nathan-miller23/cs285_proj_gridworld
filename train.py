import argparse
import math
import numpy as np
import os
import pickle
import torch
import torch.optim as optim
import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime

from utils import load
from rl import tabular_learning
from generate_data import DATA_DIR
from agents import AgentFromTorch

CURR_DIR = os.path.abspath(os.path.dirname(__file__))

USE_CUDA = False

class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)

class Net(nn.Module):

    def __init__(self, in_shape, out_size, conv_arch, filter_size, stride, fc_arch, **kwargs):
        super().__init__()
        self.in_shape = in_shape
        self.out_size = out_size
        self.conv_arch = conv_arch
        self.fc_arch = fc_arch
        self.filter_size = filter_size
        self.stride = stride

        pad = self.filter_size // 2
        padding = (pad, pad)

        layers = []
        in_h, in_w, in_channels = in_shape
        if conv_arch:
            layers.append(nn.Conv2d(in_shape[2], self.conv_arch[0], self.filter_size, self.stride, padding=padding))
            in_channels = self.conv_arch[0]
            in_h = math.ceil(in_h / 2)
            in_w = math.ceil(in_w / 2)

        for channels in conv_arch:
            layers.append(nn.Conv2d(in_channels, channels, self.filter_size, self.stride, padding=padding))
            in_channels = channels
            in_h = math.ceil(in_h / 2)
            in_w = math.ceil(in_w / 2)

        layers.append(Flatten())

        in_features = in_channels * in_h * in_w

        for hidden_size in fc_arch:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        layers.append(nn.Linear(in_features, out_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data(data_path, dataset_size):
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    states = data_dict['states'][:dataset_size]
    actions = data_dict['actions'][:dataset_size]

    return states, actions


def strategic_advantage_weighted_cross_entropy(logits, labels, states, A_strat):
    weights = []
    for state in states:
        weights.append(A_strat(state.numpy()))
    logprobs = torch.gather(nn.LogSoftmax(1)(logits), 1, labels.unsqueeze(1))
    
    weights = torch.tensor(weights).unsqueeze(1)
    if (USE_CUDA):
        weights = weights.cuda()
    losses = weights * logprobs
    return -torch.sum(losses) / torch.sum(weights)


def train(model, X, Y, train_params, A_strat, env):
    weight_type = "a_strat" if A_strat else "vanilla"
    logdir = os.path.join(train_params['logdir'], "{}_{}_{}".format(weight_type, train_params['experiment_name'], datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    writer = SummaryWriter(log_dir=logdir)
    X = torch.Tensor(X)
    Y = torch.Tensor(Y)
    full_dataset = TensorDataset(X, Y)

    train_size = int(train_params['train_size'] * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, test_size])
    train_loader, val_loader = DataLoader(train_dataset, batch_size=train_params['batch_size']), DataLoader(val_dataset)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=train_params['learning_rate'])
    global device
    global USE_CUDA
    for epoch in range(train_params['num_epochs']):  # loop over the dataset multiple times
        train_loss = 0.0
        for i, data in tqdm.tqdm(enumerate(train_loader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            states, labels = data
            labels = labels.type(torch.long)
            inputs = states.permute(0, 3, 1, 2)

            # zero the parameter gradients
            optimizer.zero_grad()
            if (USE_CUDA):
                inputs = inputs.cuda()
                labels = labels.cuda()

            # forward + backward + optimize
            outputs = model(inputs)

            if (USE_CUDA):
                outputs=outputs.cuda()

            if train_params['strategic_advantage']:
                loss = strategic_advantage_weighted_cross_entropy(outputs, labels, states, A_strat)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                states, labels = data
                labels = labels.type(torch.long)
                inputs = states.permute(0, 3, 1, 2)
                if (USE_CUDA):
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = model(inputs)
                if (USE_CUDA):
                    outputs = outputs.cuda()
                
                if train_params['strategic_advantage']:
                    loss = strategic_advantage_weighted_cross_entropy(outputs, labels, states, A_strat).item()
                else:
                    loss = criterion(outputs, labels)
                val_loss += loss
        
        val_rewards = []
        eval_agent = AgentFromTorch(model, use_cuda = True)
        for _ in range(train_params['num_validation_episodes']):
            obs = env.reset()
            done=False
            val_reward = 0.0
            while not done:
                action = eval_agent.action(obs)
                obs, reward, done, _ = env.step(action)
                val_reward += reward
            val_rewards.append(val_reward)
        model.train()


        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        val_reward_mean = np.mean(val_rewards)
        val_reward_std = np.std(val_rewards)
        writer.add_scalar("Loss/val_loss", val_loss, epoch)
        writer.add_scalar("Loss/train_loss", train_loss, epoch)
        writer.add_scalar("Reward/val_reward_mean", val_reward_mean, epoch)
        writer.add_scalar("Reward/val_reward_std", val_reward_std, epoch)

        if epoch % train_params['model_save_freq'] == 0:
            torch.save(model.state_dict(), os.path.join(logdir, "checkpoint_{}".format(epoch)))
        print("")
        print("Losses at end of Epoch {}\nTrain: {}\nVal: {}".format(epoch, train_loss, val_loss))
        print("Sample Reward at end of Epoch {}\nReward: {}".format(epoch, val_reward_mean))
        print("")


def main(params):
    set_seed(params['seed'])

    # Paths Parsing
    exp_data_dir = os.path.join(params['data_dir'], params['infile_name'])
    data_load_loc = os.path.join(exp_data_dir, 'data.pkl')
    env_load_loc = os.path.join(exp_data_dir, 'env.pkl')
    agent_load_loc = os.path.join(exp_data_dir, 'agent.pkl')

    # Load our data
    X, Y = load_data(data_load_loc, params['dataset_size'])
    env = load(env_load_loc)
    agent = load(agent_load_loc)
    in_shape = X[0].shape
    out_size = len(np.unique(Y))

    params['in_shape'] = in_shape
    params['out_size'] = out_size

    global USE_CUDA

    
        

    # Build network
    model = Net(**params)
    if (params['cuda'] and torch.cuda.is_available()):
        USE_CUDA = True
        print("using cuda")
        model.cuda()
        agent.use_cuda = True

    A_strat = None
    if params['strategic_advantage']:
        _, _, A_strat = tabular_learning(env, agent, gamma=0.9)

    train(model, X, Y, params, A_strat, env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-dp", type=str, default=DATA_DIR)
    parser.add_argument('--infile_name', '-i', type=str, default="my_exp")
    parser.add_argument('--fc_arch','-fc', nargs='+', type=int, default=[100])
    parser.add_argument('--conv_arch', '-cv', nargs='+', type=int, default=[8, 16])
    parser.add_argument('--num_epochs', '-e', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=2000)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--filter_size', '-fs', type=int, default=3)
    parser.add_argument('--stride', '-st', type=int, default=2)
    parser.add_argument('--train_size', '-ts', type=float, default=0.8)
    parser.add_argument('--dataset_size', '-n', type=int, default=1000)
    parser.add_argument('--num_validation_episodes', '-nv', type=int, default=100)
    parser.add_argument('--strategic_advantage', '-adv', action='store_true')
    parser.add_argument('--online_q_learning', '-on', action="store_true")
    parser.add_argument('--logdir', '-ld', type=str, default=os.path.join(CURR_DIR, 'runs'))
    parser.add_argument('--experiment_name', '-exp', type=str, required=True)
    parser.add_argument('--model_save_freq', '-sf', type=int, default=5)
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--cuda', '-c', action='store_true')

    params = vars(parser.parse_args())
    main(params)
    
