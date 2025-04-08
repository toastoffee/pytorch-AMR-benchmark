import torch
from torch import nn

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    extention = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * BasicBlock.extention, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_channels * BasicBlock.extention)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.extention * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BasicBlock.extention, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * BasicBlock.extention)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class grrNet(nn.Module):

    def __init__(self, num_class):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=7, stride=1, padding=3)

        self.block1 = BasicBlock(in_channels=16, out_channels=32)
        self.block2 = BasicBlock(in_channels=32, out_channels=32)
        self.block3 = BasicBlock(in_channels=32, out_channels=32)
        self.block4 = BasicBlock(in_channels=32, out_channels=32)

        self.block5 = BasicBlock(in_channels=32, out_channels=64)
        self.block6 = BasicBlock(in_channels=64, out_channels=64)
        self.block7 = BasicBlock(in_channels=64, out_channels=64)
        self.block8 = BasicBlock(in_channels=64, out_channels=64)

        self.gru = nn.GRU(input_size=128, hidden_size=64, num_layers=1)

        self.fc = nn.Linear(in_features=4096, out_features=num_class)

    def forward(self, x):
        x = self.conv1d(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        output, h = self.gru(x)

        reshaped = output.reshape(output.shape[0], -1)

        output = self.fc(reshaped)

        return output

if __name__ == '__main__':
    encoder = grrNet(num_class=11)

    sgn = torch.randn((64, 2, 128))

    # block = BasicBlock(in_channels=16, out_channels=16)

    sgn = encoder(sgn)
    # sgn = block(sgn)

    print(sgn.shape)
