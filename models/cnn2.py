import torch
import numpy as np
from torch import nn

class CNN2(nn.Module):
    def __init__(self, num_classes):
        super(CNN2, self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(1, 256, kernel_size=(1,7), stride=1,padding=(0,3) ,bias=True),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=(1, 2)),
                                # nn.Dropout(0.2)
                                )

        self.conv2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=(1,7), stride=1,padding=(0,3) ,bias=True),
                                   nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=(1, 2)),
                                   # nn.Dropout(0.2)
                                )
        self.conv3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=(1,7), stride=1,padding=(0,3) ,bias=True),
                                   nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=(1, 2)),
                                   # nn.Dropout(0.2)
                                   )
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(1,7), stride=1,padding=(0,3) ,bias=True),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(1, 2)),
                                   # nn.Dropout(0.2)
                                   )

        self.fc1 = nn.Sequential(nn.Linear(in_features=1024, out_features=128),
                                 nn.ReLU()
                                )
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, y):
        y = y.unsqueeze(1)
        y=self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.fc2(y)
        return y