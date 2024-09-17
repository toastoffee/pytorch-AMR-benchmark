import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Fits(nn.Module):

    # FITS: Frequency Interpolation Time Series Forecasting

    def __init__(self, seq_len, pred_len, individual, enc_in, cut_freq):
        super(Fits, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.channels = enc_in

        # Decompsition Kernel Size
        kernel_size = 25
        self.dominance_freq = cut_freq  # 720/24
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(
                    nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(torch.cfloat))

        else:
            self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(
                torch.cfloat)  # complex layer for frequency upcampling]
        # configs.pred_len=configs.seq_len+configs.pred_len
        # #self.Dlinear=DLinear.Model(configs)
        # configs.pred_len=self.pred_len

    def forward(self, x):

        x = x.permute(0, 2, 1)

        # RIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        # print(x_var)
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:, self.dominance_freq:] = 0
        # low_x=torch.fft.irfft(low_specx, dim=1)

        # no cut off
        low_specx = low_specx[:, 0:self.dominance_freq, :]

        # print(low_specx.permute(0,2,1))
        if self.individual:
            low_specxy_ = torch.zeros(
                [low_specx.size(0), int(self.dominance_freq * self.length_ratio), low_specx.size(2)],
                dtype=low_specx.dtype).to(low_specx.device)
            for i in range(self.channels):
                low_specxy_[:, :, i] = self.freq_upsampler[i](low_specx[:, :, i].permute(0, 1)).permute(0, 1)
        else:
            # print(low_specx.permute(0,2,1).size())
            low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(0, 2, 1)
        # print(low_specxy_)
        low_specxy = torch.zeros(
            [low_specxy_.size(0), int((self.seq_len + self.pred_len) / 2 + 1), low_specxy_.size(2)],
            dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:, 0:low_specxy_.size(1), :] = low_specxy_
        low_xy = torch.fft.irfft(low_specxy, dim=1)
        low_xy = low_xy * self.length_ratio  # compemsate the length change
        # dom_x=x-low_x

        # dom_xy=self.Dlinear(dom_x)
        # xy=(low_xy+dom_xy) * torch.sqrt(x_var) +x_mean # REVERSE RIN
        xy = (low_xy) * torch.sqrt(x_var) + x_mean
        xy = xy.permute(0, 2, 1)

        return xy, low_xy * torch.sqrt(x_var)

class FitsWithCNN2(nn.Module):

    # FITS: Frequency Interpolation Time Series Forecasting

    def __init__(self, seq_len, pred_len, individual, enc_in, cut_freq, num_classes = 11):
        super(FitsWithCNN2, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.channels = enc_in

        # Decompsition Kernel Size
        kernel_size = 25
        self.dominance_freq = cut_freq  # 720/24
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(
                    nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(torch.cfloat))

        else:
            self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(
                torch.cfloat)  # complex layer for frequency upcampling]
        # configs.pred_len=configs.seq_len+configs.pred_len
        # #self.Dlinear=DLinear.Model(configs)
        # configs.pred_len=self.pred_len

        self.conv1 = nn.Sequential(nn.Conv2d(1, 256, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=True),
                                   #                                  nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(1, 2)),
                                   # nn.Dropout(0.2)
                                   )

        self.conv2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=True),
                                   #                                    nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(1, 2)),
                                   # nn.Dropout(0.2)
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=True),
                                   #                                    nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(1, 2)),
                                   # nn.Dropout(0.2)
                                   )
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=True),
                                   #                                    nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(1, 2)),
                                   # nn.Dropout(0.2)
                                   )

        self.fc1 = nn.Sequential(nn.Linear(in_features=1024, out_features=128),
                                 nn.ReLU()
                                 )
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):

        x = x.permute(0, 2, 1)

        # RIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        # print(x_var)
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:, self.dominance_freq:] = 0
        # low_x=torch.fft.irfft(low_specx, dim=1)

        # no cut off
        low_specx = low_specx[:, 0:self.dominance_freq, :]

        # print(low_specx.permute(0,2,1))
        if self.individual:
            low_specxy_ = torch.zeros(
                [low_specx.size(0), int(self.dominance_freq * self.length_ratio), low_specx.size(2)],
                dtype=low_specx.dtype).to(low_specx.device)
            for i in range(self.channels):
                low_specxy_[:, :, i] = self.freq_upsampler[i](low_specx[:, :, i].permute(0, 1)).permute(0, 1)
        else:
            # print(low_specx.permute(0,2,1).size())
            low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(0, 2, 1)
        # print(low_specxy_)
        low_specxy = torch.zeros(
            [low_specxy_.size(0), int((self.seq_len + self.pred_len) / 2 + 1), low_specxy_.size(2)],
            dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:, 0:low_specxy_.size(1), :] = low_specxy_
        low_xy = torch.fft.irfft(low_specxy, dim=1)
        low_xy = low_xy * self.length_ratio  # compemsate the length change
        # dom_x=x-low_x

        # dom_xy=self.Dlinear(dom_x)
        # xy=(low_xy+dom_xy) * torch.sqrt(x_var) +x_mean # REVERSE RIN
        xy = (low_xy) * torch.sqrt(x_var) + x_mean
        xy = xy.permute(0, 2, 1)

        y = xy.unsqueeze(1)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.fc2(y)

        return y


if __name__ == "__main__":
    net = FitsWithCNN2(128, 0, False, 2, 60)

    sgn = torch.randn((3, 2, 128))
    sgn = net(sgn)
    print(sgn.shape)