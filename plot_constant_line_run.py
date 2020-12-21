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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--probability_zero', '-p0', type=float, default=0.5)
    parser.add_argument('--probability_one', '-p1', type=float, default=0.5)
    parser.add_argument('--experiment_name', '-o', type=str, required=True)
    parser.add_argument('--constant', '-c', type=float, default=None)
    args = parser.parse_args()
    logdir = os.path.join(LOG_DIR, args.experiment_name)
    writer = SummaryWriter(log_dir=logdir)
    minepoch = 0
    maxepoch = 300
    #Loading data, env, agent
    print("starting logging")
    p0, p1 = args.probability_zero, args.probability_one
    mean = args.constant if args.constant else 10*(p1 - p0) / (p1+p0)
    print(mean)
    for i in range(maxepoch):
        writer.add_scalar("Reward/val_reward_mean", mean, i)
    print("finished logging")





