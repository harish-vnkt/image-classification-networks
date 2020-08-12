import torch
import torch.nn as nn
import torch.nn.functional as F


class Layer(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, last=False):

        super(Layer, self).__init__()

        self.last = last
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):

        if not self.last:
            return F.relu(self.batch_norm(self.conv(x)))
        else:
            return self.batch_norm(self.conv(x))


class ResidualBlock(nn.Module):

    def __init__(self, n, in_channels, out_channels, downsample=False):

        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample

        layers = [Layer(in_channels=self.in_channels, out_channels=self.out_channels, stride=2 if downsample else 1)]
        for _ in range(1, 2 * n - 1):
            layers.append(Layer(in_channels=out_channels, out_channels=out_channels))
        layers.append(Layer(in_channels=out_channels, out_channels=out_channels, last=True))

        self.block = nn.Sequential(*layers)

        self.skip_connection = nn.Sequential()
        if downsample:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(num_features=self.out_channels)
            )

    def forward(self, x):

        out = self.block(x)
        out += self.skip_connection(x)
        return F.relu(out)


class Resnet(nn.Module):

    def __init__(self, n):

        super(Resnet, self).__init__()

        self.n = n
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=16)

        self.in_channels = 16
        self.out_channels = 16

        layers = [ResidualBlock(self.n, self.in_channels, self.out_channels)]
        self.in_channels = self.out_channels
        self.out_channels = self.in_channels * 2

        for _ in range(1, n):
            layers.append(ResidualBlock(self.n, self.in_channels, self.out_channels, downsample=True))
            self.in_channels = self.out_channels
            self.out_channels = self.in_channels * 2

        self.residual_blocks = nn.Sequential(*layers)
        self.linear = nn.Linear(in_features=64, out_features=100)

    def forward(self, x):

        out = F.relu(self.batch_norm(self.conv(x)))
        out = self.residual_blocks(out)
        out = F.max_pool2d(out, 8)
        out = out.view(64, -1)
        out = self.linear(out)
        return out


