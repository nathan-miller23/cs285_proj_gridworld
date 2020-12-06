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

from generate_data import load
from rl import tabular_learning

CURR_DIR = os.path.abspath('.')


class Net(nn.Module):

    def __init__(self, in_shape, out_size, conv_arch, filter_size, stride, fc_arch):
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
        layers.append(nn.Conv2d(in_shape[2], self.conv_arch[0], self.filter_size, self.stride, padding=padding))
        in_channels = self.conv_arch[0]
        in_h, in_w = in_shape[:2]
        in_h = math.ceil(in_h / 2)
        in_w = math.ceil(in_w / 2)

        for channels in conv_arch:
            layers.append(nn.Conv2d(in_channels, channels, self.filter_size, self.stride, padding=padding))
            in_channels = channels
            in_h = math.ceil(in_h / 2)
            in_w = math.ceil(in_w / 2)

        layers.append(nn.Flatten())

        in_features = in_channels * in_h * in_w

        for hidden_size in fc_arch:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        layers.append(nn.Linear(in_features, out_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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
    weights = F.softmax(torch.tensor(weights) + 10).unsqueeze(1) # TODO for balancing
    logprobs = torch.gather(nn.LogSoftmax(1)(logits), 1, labels.unsqueeze(1))
    losses = weights * logprobs
    print(losses.shape, weights.shape, logprobs.shape)
    return -torch.sum(losses)


def train(model, X, Y, train_params, A_strat):
    X = torch.Tensor(X)
    Y = torch.Tensor(Y)
    full_dataset = TensorDataset(X, Y)

    train_size = int(train_params['train_size'] * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, test_size])
    train_loader, val_loader = DataLoader(train_dataset, batch_size=train_params['batch_size']), DataLoader(val_dataset)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=train_params['lr'])

    for epoch in range(train_params['num_epochs']):  # loop over the dataset multiple times
        train_loss = 0.0
        for i, data in tqdm.tqdm(enumerate(train_loader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            states, labels = data
            labels = labels.type(torch.long)
            inputs = states.permute(0, 3, 1, 2)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            if train_params['strategic_advantage']:
                loss = strategic_advantage_weighted_cross_entropy(outputs, labels, states, A_strat)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()

        val_loss = 0.0
        for data in val_loader:
            inputs, labels = data
            inputs = inputs.permute(0, 3, 1, 2)
            labels = labels.type(torch.long)

            # no grad
            outputs = model(inputs)
            loss = criterion(outputs, labels).item()
            val_loss += loss
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)

        print("")
        print("Losses at end of Epoch {}\nTrain: {}\nVal: {}".format(epoch, train_loss, val_loss))
        print("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-dp", type=str, default=os.path.join(CURR_DIR, 'data.pkl'))
    parser.add_argument('--agent_path', '-ap', type=str, default=os.path.join(CURR_DIR, 'agent.pkl'))
    parser.add_argument('--env_path', '-ep', type=str, default=os.path.join(CURR_DIR, 'env.pkl'))
    parser.add_argument('--fc_arch', '-fc', nargs='+', type=int, default=[100])
    parser.add_argument('--conv_arch', '-cv', nargs='+', type=int, default=[8, 16])
    parser.add_argument('--num_epochs', '-e', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=2000)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--filter_size', '-fs', type=int, default=3)
    parser.add_argument('--stride', '-s', type=int, default=2)
    parser.add_argument('--train_size', '-ts', type=float, default=0.8)
    parser.add_argument('--dataset_size', '-n', type=int, default=1000)
    parser.add_argument('--strategic_advantage', '-adv', action='store_true')
    parser.add_argument('--online_q_learning', '-on', action="store_true")

    args = parser.parse_args()

    X, Y = load_data(args.data_path, args.dataset_size)

    in_shape = X[0].shape
    out_size = len(np.unique(Y))

    model_params = {
        "fc_arch": args.fc_arch,
        "conv_arch": args.conv_arch,
        "filter_size": args.filter_size,
        "stride": args.stride,
        "in_shape": in_shape,
        "out_size": out_size
    }

    training_params = {
        "lr": args.learning_rate,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "train_size": args.train_size,
        "strategic_advantage": args.strategic_advantage
    }

    model = Net(**model_params)

    A_strat = None

    if args.strategic_advantage:
        env = load(args.env_path)
        agent = load(args.agent_path)
        _, _, A_strat = tabular_learning(env, agent, gamma=0.9)

    train(model, X, Y, training_params, A_strat)


if __name__ == '__main__':
    main()
