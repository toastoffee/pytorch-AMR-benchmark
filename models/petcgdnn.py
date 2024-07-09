import torch
import numpy as np
from torch import nn
import torch.fft
import math

class PETCGDNN(nn.Module):
    def __init__(self, num_classes):
        super(PETCGDNN, self).__init__()
        self.fc1=nn.Linear(256,1)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 75, kernel_size=(2, 8), stride=1, padding=(0, 0), bias=True),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(75),
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(75, 25, kernel_size=(1, 5), stride=1, padding=(0, 0), bias=True),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(25),
                                   )
        self.gru=nn.GRU(input_size=25, hidden_size=128,batch_first=True)

        self.fc2 = nn.Sequential(nn.Linear(14976,1024),
                              nn.ReLU(),
                              nn.Dropout(0.5),
                              nn.Linear(1024, num_classes))
    def forward(self, y):
        y = torch.squeeze(y, 1)
        y = y.unsqueeze(1)
        input1=y[:,:,0,:]
        input2 = y[:, :, 1, :]
        y=y.view(y.size(0), -1)
        y=self.fc1(y)
        cos1=torch.cos(y)
        cos1 = cos1.unsqueeze(1)
        sin1=torch.sin(y)
        sin1 = sin1.unsqueeze(1)
        x11=torch.matmul(cos1,input1)
        x12 = torch.matmul(sin1, input2)
        x21 = torch.matmul(cos1, input2)
        x22 = torch.matmul(sin1, input1)
        y1=x11+x12
        y2 = x21 - x22
        x3=torch.cat((y1,y2),dim=1)
        x3 = x3.unsqueeze(1)
        x3=self.conv1(x3)
        x3 = self.conv2(x3)
        x3 = x3.squeeze(2)
        x3 = x3.transpose(1, 2)
        self.gru.flatten_parameters()
        x3, (hidden)=self.gru(x3)
        # x3 = x3[:, -1, :]
        x3 = x3.contiguous().view(x3.size()[0], -1)
        y=self.fc2(x3)
        return y

class PETCGDNN_encoder(nn.Module):
    def __init__(self, num_classes):
        super(PETCGDNN_encoder, self).__init__()
        self.fc1 = nn.Linear(256, 1)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 75, kernel_size=(2, 8), stride=1, padding=(0, 0), bias=True),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(75),
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(75, 25, kernel_size=(1, 5), stride=1, padding=(0, 0), bias=True),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(25),
                                   )
        self.gru = nn.GRU(input_size=25, hidden_size=128, batch_first=True)

        self.fc2 = nn.Sequential(nn.Linear(14976, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(1024, num_classes))

    def forward(self, y):
        y = torch.squeeze(y, 1)
        y = y.unsqueeze(1)
        input1=y[:,:,0,:]
        input2 = y[:, :, 1, :]
        y=y.view(y.size(0), -1)
        y=self.fc1(y)
        cos1=torch.cos(y)
        cos1 = cos1.unsqueeze(1)
        sin1=torch.sin(y)
        sin1 = sin1.unsqueeze(1)
        x11=torch.matmul(cos1,input1)
        x12 = torch.matmul(sin1, input2)
        x21 = torch.matmul(cos1, input2)
        x22 = torch.matmul(sin1, input1)
        y1=x11+x12
        y2 = x21 - x22
        x3=torch.cat((y1,y2),dim=1)
        x3 = x3.unsqueeze(1)
        x3=self.conv1(x3)
        x3 = self.conv2(x3)
        x3 = x3.squeeze(2)
        x3 = x3.transpose(1, 2)
        self.gru.flatten_parameters()
        x3, (hidden)=self.gru(x3)
        # x3 = x3[:, -1, :]
        x3 = x3.contiguous().view(x3.size()[0], -1)
        return x3

class PETCGDNN_classifier(nn.Module):
    def __init__(self, num_classes):
        super(PETCGDNN_classifier, self).__init__()
        self.fc1=nn.Linear(256,1)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 75, kernel_size=(2, 8), stride=1, padding=(0, 0), bias=True),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(75),
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(75, 25, kernel_size=(1, 5), stride=1, padding=(0, 0), bias=True),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(25),
                                   )
        self.gru=nn.GRU(input_size=25, hidden_size=128,batch_first=True)

        self.fc2 = nn.Sequential(nn.Linear(14976,1024),
                              nn.ReLU(),
                              nn.Dropout(0.5),
                              nn.Linear(1024, num_classes))
    def forward(self, y):

        y = self.fc2(y)
        return y


if __name__ == '__main__':
    encoder = PETCGDNN_encoder(num_classes=11)
    classifier = PETCGDNN_classifier(num_classes=11)
    whole = PETCGDNN(num_classes=11)
    sgn = torch.randn((64, 2, 128))

    sgn = encoder(sgn)
    print(sgn.shape)
    # #
    sgn = classifier(sgn)
    print(sgn.shape)
