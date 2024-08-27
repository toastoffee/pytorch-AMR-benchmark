import torch
from torch import nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    extention = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * BasicBlock.extention, kernel_size=3, padding=1, bias=False),
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

class Bottleneck(nn.Module):
    #每个stage维度中扩展的倍数
    extention=4
    def __init__(self,inplanes,planes,stride,downsample=None):
        '''

        :param inplanes: 输入block的之前的通道数
        :param planes: 在block中间处理的时候的通道数
                planes*self.extention:输出的维度
        :param stride:
        :param downsample:
        '''
        super(Bottleneck, self).__init__()

        self.conv1=nn.Conv1d(inplanes,planes,kernel_size=1,stride=stride,bias=False)
        self.bn1=nn.BatchNorm1d(planes)

        self.conv2=nn.Conv1d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm1d(planes)

        self.conv3=nn.Conv1d(planes,planes*self.extention,kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm1d(planes*self.extention)

        self.relu=nn.ReLU(inplace=True)

        #判断残差有没有卷积
        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        #参差数据
        residual=x

        #卷积操作
        out1=self.conv1(x)
        out2=self.bn1(out1)
        out3=self.relu(out2)

        out4=self.conv2(out3)
        out5=self.bn2(out4)
        out6=self.relu(out5)

        out7=self.conv3(out6)
        out8=self.bn3(out7)
        out9=self.relu(out8)

        #是否直连（如果Indentity blobk就是直连；如果Conv2 Block就需要对残差边就行卷积，改变通道数和size
        if self.downsample is not None:
            residual=self.downsample(x)

        #将残差部分和卷积部分相加
        out10=out9+residual
        out11=self.relu(out10)

        return out11


class ResNet1d(nn.Module):
    def __init__(self, numclass,block=Bottleneck,layers=[3,4,6,3]):
        #inplane=当前的fm的通道数
        self.inplane=64
        super(ResNet1d, self).__init__()

        #参数
        self.block=block
        self.layers=layers

        #stem的网络层
        self.conv1=nn.Conv1d(2,self.inplane,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm1d(self.inplane)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool1d(kernel_size=3,stride=2,padding=1)

        #64,128,256,512指的是扩大4倍之前的维度，即Identity Block中间的维度
        self.stage1=self.make_layer(self.block,64,layers[0],stride=1)
        self.stage2=self.make_layer(self.block,128,layers[1],stride=2)
        self.stage3=self.make_layer(self.block,256,layers[2],stride=2)
        self.stage4=self.make_layer(self.block,512,layers[3],stride=2)
        self.gap=nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(2048, numclass)

        )

    def forward(self, y):
        #stem部分：conv+bn+maxpool
        out=self.conv1(y)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.maxpool(out)

        #block部分
        out=self.stage1(out)
        out=self.stage2(out)
        out=self.stage3(out)
        out=self.stage4(out)
        out=self.gap(out)
        out=out.view(out.size(0), -1)
        out=self.classifier(out)

        return out

    def make_layer(self,block,plane,block_num,stride=1):
        '''
        :param block: block模板
        :param plane: 每个模块中间运算的维度，一般等于输出维度/4
        :param block_num: 重复次数
        :param stride: 步长
        :return:
        '''
        block_list=[]
        #先计算要不要加downsample
        downsample=None
        if(stride!=1 or self.inplane!=plane*block.extention):
            downsample=nn.Sequential(
                nn.Conv1d(self.inplane,plane*block.extention,stride=stride,kernel_size=1,bias=False),
                nn.BatchNorm1d(plane*block.extention)
            )

        # Conv Block输入和输出的维度（通道数和size）是不一样的，所以不能连续串联，他的作用是改变网络的维度
        # Identity Block 输入维度和输出（通道数和size）相同，可以直接串联，用于加深网络
        #Conv_block
        conv_block=block(self.inplane,plane,stride=stride,downsample=downsample)
        block_list.append(conv_block)
        self.inplane=plane*block.extention

        #Identity Block
        for i in range(1,block_num):
            block_list.append(block(self.inplane,plane,stride=1))

        return nn.Sequential(*block_list)

def resnet10(num_class):
    """ return a ResNet 10 object
    """
    return ResNet1d(numclass=num_class, block=BasicBlock, layers=[1, 1, 1, 1])

def resnet18(num_class):
    """ return a ResNet 18 object
    """
    return ResNet1d(numclass=num_class, block=BasicBlock, layers=[2, 2, 2, 2])


def resnet34(num_class):
    """ return a ResNet 34 object
    """
    return ResNet1d(numclass=num_class, block=Bottleneck, layers=[3, 4, 6, 3])


def resnet50(num_class):
    """ return a ResNet 50 object
    """
    return ResNet1d(numclass=num_class, block=BasicBlock, layers=[3, 4, 6, 3])


def resnet101(num_class):
    """ return a ResNet 101 object
    """
    return ResNet1d(numclass=num_class, block=Bottleneck, layers=[3, 4, 23, 3])


def resnet152(num_class):
    """ return a ResNet 152 object
    """
    return ResNet1d(numclass=num_class, block=Bottleneck, layers=[3, 8, 36, 3])


if __name__ == '__main__':
    encoder = resnet152(num_class=11)

    sgn = torch.randn((64, 2, 128))

    sgn = encoder(sgn)
    print(sgn.shape)