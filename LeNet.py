import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.linear = nn.Linear(in_features=1 * 1 * 120, out_features=100)

    def forward(self, x):

        output = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        output = F.max_pool2d(torch.relu(self.conv2(output)), 2)
        output = torch.relu(self.conv3(output))
        final = torch.relu(self.linear(output))
        return final

