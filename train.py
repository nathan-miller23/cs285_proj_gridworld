import torch, pickle, tqdm, argparse, os, math

import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.utils.data import random_split, TensorDataset, DataLoader
from torch import nn

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

def load_data(data_path):
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    states = data_dict['states']
    actions = data_dict['actions']

    return states, actions

def train(model, X, Y, train_params):
    X = torch.Tensor(X)
    Y = torch.Tensor(Y)
    full_dataset = TensorDataset(X,Y)

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
            inputs, labels = data
            labels = labels.type(torch.long)
            inputs = inputs.permute(0, 3, 1, 2)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
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
    parser.add_argument("--data_path", "-dp", type=str, default=os.path.join(CURR_DIR, 'out.pkl'))
    parser.add_argument('--fc_arch','-fc', nargs='+', type=int, default=[100])
    parser.add_argument('--conv_arch', '-cv', nargs='+', type=int, default=[8, 16])
    parser.add_argument('--num_epochs', '-e', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=2000)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--filter_size', '-fs', type=int, default=3)
    parser.add_argument('--stride', '-s', type=int, default=2)
    parser.add_argument('--train_size', '-ts', type=float, default=0.8)

    args = parser.parse_args()

    X, Y = load_data(args.data_path)
    
    in_shape = X[0].shape
    out_size = len(np.unique(Y))

    model_params = {
        "fc_arch" : args.fc_arch,
        "conv_arch" : args.conv_arch,
        "filter_size" : args.filter_size,
        "stride" : args.stride,
        "in_shape" : in_shape,
        "out_size" : out_size
    }

    training_params = {
        "lr" : args.learning_rate,
        "batch_size" : args.batch_size,
        "num_epochs" : args.num_epochs,
        "train_size" : args.train_size
    }

    model = Net(**model_params)
    
    train(model, X, Y, training_params)

if __name__ == '__main__':
    main()
    