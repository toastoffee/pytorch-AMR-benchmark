import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN, self).__init__()

        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(1, 8),
                                              stride=1, padding=(0, 0), bias=True),
                                    nn.ReLU())
        self.dropout_1 = nn.Dropout(0.5)

        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(2, 8),
                                              stride=1, padding="valid", bias=True),
                                    nn.ReLU())
        self.dropout_2 = nn.Dropout(0.5)

        self.dense_1 = nn.Sequential(nn.Linear(in_features=6050, out_features=256),
                                     nn.ReLU())
        self.dropout_3 = nn.Dropout(0.5)

        self.dense_2 = nn.Sequential(nn.Linear(in_features=256, out_features=num_classes),
                                     nn.Softmax(dim=1))

    def forward(self, y: torch.Tensor):
        y = y.unsqueeze(dim=1)                                  # (N, 2, 128) => (N, 1, 2, 128)

        y = F.pad(y, (0, 7, 0, 0), mode='constant', value=0)
        y = self.conv_1(y)                                      # (N, 1, 2, 128) => (N, 50, 2, 128)
        y = self.dropout_1(y)

        y = self.conv_2(y)                                      # (N, 50, 2, 128) => (N, 50, 1, 121)
        y = self.dropout_2(y)

        y = y.view(y.size(0), -1)                               # Flatten (N, 6050)

        y = self.dense_1(y)                                     # (N, 256)
        y = self.dropout_3(y)

        y = self.dense_2(y)                                     # (N, num_classes)

        return y


if __name__ == "__main__":
    sgn = torch.randn((64, 2, 128))

    net = CNN(num_classes=11)

    sgn = net(sgn)

    print(sgn.shape)
