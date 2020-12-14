import os
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
from train import *

seeds = [1, 2, 3]
dataset_sizes = [100, 250, 500, 1000, 2000, 4000]
cmd_string = "python generate_data.py --save_agent --save_environment --epsilon 0.4 --delta 0.3"
for seed in seeds:
    for size in dataset_sizes:
        logdir = os.path.join(LOG_DIR, "{}_{}".format("const"+ "{" + str(size) + "}" + "s" + str(seed), datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
        print("Seed:", seed, "Size:", size)
        print("generating data")
        os.system(cmd_string)
        print("data generated")
        set_seed(seed)
        exp_data_dir = os.path.join(DATA_DIR, "out")
        data_load_loc = os.path.join(exp_data_dir, 'data.pkl')
        writer = SummaryWriter(log_dir=logdir)
        minepoch = 0
        maxepoch = 300
        #Loading data, env, agent
        print("Loading data, env, agent")
        X, Y, probs = load_data(data_load_loc, size, False, False, True)
        print(probs)
        print("Loaded data")
        mean = 10*(probs[1] - probs[0]) / (probs[1]+probs[0])
        writer.add_scalar("Reward/val_reward_mean", mean, minepoch)
        writer.add_scalar("Reward/val_reward_mean", mean, maxepoch)
        print("finished logging")





