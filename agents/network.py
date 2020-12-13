from torch import nn
import math


class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)

class Net(nn.Module):
    def __init__(self, in_shape, out_size, conv_arch=[8, 8], filter_size=3, stride=1, fc_arch=[64, 64], **kwargs):
        super().__init__()
        self.in_shape = in_shape
        self.out_size = out_size
        self.conv_arch = conv_arch if conv_arch else []
        self.fc_arch = fc_arch if fc_arch else []
        self.filter_size = filter_size
        self.stride = stride

        pad = self.filter_size // 2
        padding = (pad, pad)

        layers = []
        in_h, in_w, in_channels = self.in_shape
        if conv_arch:
            layers.append(nn.Conv2d(self.in_shape[2], self.conv_arch[0], self.filter_size, self.stride, padding=padding))
            in_channels = self.conv_arch[0]
            in_h = math.ceil(in_h / stride)
            in_w = math.ceil(in_w / stride)

        for channels in self.conv_arch:
            layers.append(nn.Conv2d(in_channels, channels, self.filter_size, self.stride, padding=padding))
            in_channels = channels
            in_h = math.ceil(in_h / stride)
            in_w = math.ceil(in_w / stride)

        layers.append(Flatten())

        in_features = in_channels * in_h * in_w

        for hidden_size in self.fc_arch:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        layers.append(nn.Linear(in_features, self.out_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)