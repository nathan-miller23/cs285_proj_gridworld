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
from collections import Counter
import datetime

from agents.network import Net
from utils import load, DATA_DIR, LOG_DIR
from rl import tabular_learning, encode
from deep_rl import train_q_network
from generate_data import DATA_DIR
from agents import AgentFromTorch

USE_CUDA = False

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data(data_path, dataset_size, rbg_observations, shuffle=False, calculate_empirical_action_probs=False):
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    N = len(data_dict['states'])
    if dataset_size > N:
        raise ValueError("Attempted to parse dataset size {} but only have {} points available".format(dataset_size, N))

    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    idx = idx[:dataset_size]
    states = data_dict['states'][idx]
    actions = data_dict['actions'][idx]

    if rbg_observations:
        rbg_observations = data_dict['states_rbg'][idx]
        X, Y = (rbg_observations, states), actions
    else:
        X, Y = (states,), actions

    if calculate_empirical_action_probs:
        samples = (states, actions)
        pertinent_actions = [transition[1] for transition in zip(*samples) if encode(transition[0]) == (7, 7)]
        c = Counter()
        c.update(pertinent_actions)
        n = len(pertinent_actions)
        action_counts = [0] * 5
        for action_idx, count in c.items():
            action_counts[action_idx] = count
        action_probs = np.array(action_counts) / n
        return X, Y, action_probs
    else:
        return X, Y, None

    


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


def train(model, X, Y, train_params, A_strat, env, probs=None):
    weight_type = "a_strat" if A_strat else "vanilla"
    logdir = os.path.join(train_params['logdir'], "{}_{}_{}".format(weight_type, train_params['experiment_name'], datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    writer = SummaryWriter(log_dir=logdir)
    X_tensors = []
    for i in range(len(X)):
        X_tensors.append(torch.tensor(X[i]))
    Y_tensor = torch.tensor(Y)
    full_dataset = TensorDataset(*X_tensors, Y_tensor)

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
            observations = states = labels = None
            if train_params['rbg_observations']:
                observations, states, labels = data
            else:
                states, labels = data
                observations = states
            labels = labels.type(torch.long)
            inputs = observations.permute(0, 3, 1, 2).float()

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
                observations = states = labels = None
                if train_params['rbg_observations']:
                    observations, states, labels = data
                else:
                    states, labels = data
                    observations = states
                labels = labels.type(torch.long)
                inputs = observations.permute(0, 3, 1, 2).float()

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
        eval_agent = AgentFromTorch(model, use_cuda=USE_CUDA)
            
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

        if train_params['calculate_empirical_action_probs']:
            for i, prob in enumerate(probs):
                writer.add_scalar("Policy/action_{}_empirical_prob".format(i), prob, epoch)

        if epoch % train_params['model_save_freq'] == 0:
            torch.save((model.state_dict(), train_params), os.path.join(logdir, "checkpoint_{}".format(epoch)))
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
    rbg_env_load_loc = os.path.join(exp_data_dir, 'rbg_env.pkl')

    # Load our data
    X, Y, probs = load_data(data_load_loc, params['dataset_size'], params['rbg_observations'], params['shuffle'], params['calculate_empirical_action_probs'])
    env = load(env_load_loc)
    agent = load(agent_load_loc)
    in_shape = env.observation_space.shape
    out_size = env.action_space.n

    #if not X[0][0].shape != in_shape:
    #    raise ValueError("Env observation space shape {} does not match data observation shape {}".format(in_shape, X[0][0].shape))

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
        if params['use_deep_q_learning']:
            data = load(data_load_loc)
            A_strat = train_q_network(env, data, gamma=params['gamma'], lmbda=params['lambda'], use_cuda=USE_CUDA, dataset_size=params['dataset_size'], use_quad_net=params["use_quad_net"], max_iters=params['q_learning_iterations'])
        else:
            _, _, A_strat = tabular_learning(env, agent, gamma=params['gamma'])

    if params['rbg_observations']:
        env = load(rbg_env_load_loc)

    train(model, X, Y, params, A_strat, env, probs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-dp", type=str, default=DATA_DIR)
    parser.add_argument('--infile_name', '-i', type=str, default="my_exp")
    parser.add_argument('--fc_arch','-fc', nargs='*', type=int, default=[100])
    parser.add_argument('--conv_arch', '-cv', nargs='*', type=int, default=[8, 16])
    parser.add_argument('--num_epochs', '-e', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=2000)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--filter_size', '-fs', type=int, default=3)
    parser.add_argument('--stride', '-st', type=int, default=2)
    parser.add_argument('--train_size', '-ts', type=float, default=0.8)
    parser.add_argument('--dataset_size', '-n', type=int, default=1000)
    parser.add_argument('--num_validation_episodes', '-nv', type=int, default=100)
    parser.add_argument('--strategic_advantage', '-adv', action='store_true')
    parser.add_argument('--use_deep_q_learning', '-deep_q', action="store_true")
    parser.add_argument('--logdir', '-ld', type=str, default=LOG_DIR)
    parser.add_argument('--experiment_name', '-exp', type=str, required=True)
    parser.add_argument('--model_save_freq', '-sf', type=int, default=5)
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--cuda', '-c', action='store_true')
    parser.add_argument('--shuffle', '-shuff', action='store_true')
    parser.add_argument('--lambda', '-lam', type=float, default=1.0)
    parser.add_argument('--gamma', '-gam', type=float, default=0.95)
    parser.add_argument('--rbg_observations', '-rbg', action='store_true')
    parser.add_argument('--calculate_empirical_action_probs', '-empi', action='store_true')
    parser.add_argument('--use_quad_net', '-quad', action='store_true')
    parser.add_argument('--q_learning_iterations', '-qiters', default=5000)

    params = vars(parser.parse_args())
    main(params)
    
