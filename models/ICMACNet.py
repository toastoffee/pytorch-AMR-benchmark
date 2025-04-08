import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.fft
import math

class ICAMCNET(nn.Module):
    def __init__(self, num_classes):
        super(ICAMCNET, self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(1,64, kernel_size=(1,7), stride=1,padding=(0,3) ,bias=True),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=2),
                                 nn.BatchNorm2d(64),
                                )

        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(1,3), stride=1,padding=(0,1), bias=True),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(64),
                                )
        self.conv3 = nn.Sequential(nn.Conv2d(64,128, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=True),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=1),
                                   nn.BatchNorm2d(128),
                                   )
        self.drop1=nn.Dropout(0.4)
        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=True),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(128),
                                   )
        self.drop2 = nn.Dropout(0.4)

        self.fc1 = nn.Sequential(nn.Linear(in_features=8192, out_features=256),
                                 nn.ReLU(),
                                 )
        self.drop3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, y):
        # y = y.unsqueeze(1)
        y=self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        y=self.drop1(y)
        y=self.conv4(y)
        y=self.drop2(y)
        y=y.view(y.size(0), -1)
        y=self.fc1(y)
        y=self.drop3(y)
        g = torch.randn(y.shape[0], y.shape[1], device=y.device)
        y=y+g
        y=self.fc2(y)

        return y

    