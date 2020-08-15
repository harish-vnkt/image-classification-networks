import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)  # 64 x 32 x 32
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)  # 64 x 32 x 32

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # 128 x 16 x 16
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)  # 128 x 16 x 16

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)  # 256 x 8 x 8
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)  # 256 x 8 x 8
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)  # 256 x 8 x 8

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)  # 512 x 4 x 4
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)  # 512 x 4 x 4
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)  # 512 x 4 x 4

        # commented because network is too deep for this dataset
        # self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.linear_1 = nn.Linear(in_features=512 * 2 * 2, out_features=1024)
        self.linear_2 = nn.Linear(in_features=1024, out_features=100)

    def forward(self, x):

        output = torch.relu(self.conv1_1(x))
        output = torch.relu(self.conv1_2(output))
        output = F.max_pool2d(output, 2)

        output = torch.relu(self.conv2_1(output))
        output = torch.relu(self.conv2_2(output))
        output = F.max_pool2d(output, 2)

        output = torch.relu(self.conv3_1(output))
        output = torch.relu(self.conv3_2(output))
        output = torch.relu(self.conv3_3(output))
        output = F.max_pool2d(output, 2)

        output = torch.relu(self.conv4_1(output))
        output = torch.relu(self.conv4_2(output))
        output = torch.relu(self.conv4_3(output))
        output = F.max_pool2d(output, 2)

        output = output.view(-1, 512 * 2 * 2)

        output = torch.relu(self.linear_1(output))
        final = torch.relu(self.linear_2(output))

        return final
