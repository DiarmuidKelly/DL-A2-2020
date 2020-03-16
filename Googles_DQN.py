import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)

        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(288, 256)
        # self.fc2 = nn.Linear(256, 4)
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 64
        self.head = nn.Linear(linear_input_size, outputs)

    def weights_init_uniform_rule(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        # n = m.in_features
        # y = 1.0 / np.sqrt(n)
        # m.weight.data.uniform_(-y, y)
        # m.bias.data.fill_(0)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.fl(x)
        x = F.relu(self.fc1(x))

        # return self.head(x.view(x.size(0), -1))
        return self.head(x)
