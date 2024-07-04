from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm

class Trainer:
    def __init__(self,
                 name:                str,
                 model:               nn.Module,
                 train_dataloader:    DataLoader,
                 valid_dataloader:    DataLoader,
                 criterion:           nn.Module,
                 device:              torch.device):

        self.__name:                str             = name
        self.__model:               nn.Module       = model.to(device)
        self.__train_dataloader:    DataLoader      = train_dataloader
        self.__valid_dataloader:    DataLoader      = valid_dataloader
        self.__criterion:           nn.Module       = criterion.to(device)
        self.__device:              torch.device    = device

    def train(self,
              epochs:                       int,
              init_lr:                      float,
              lr_update_by_epochs_action:   Callable[[float, int], float]):

        lr:         float           = init_lr
        optimizer:  optim.Optimizer = optim.Adam(params=self.__model.parameters(), lr=init_lr)
        train_loss: nn.Module

        # for epoch in range(epochs):
        #     # update the lr
        #     lr = lr_update_by_epochs_action(lr, epoch)
        #     optimizer = optim.Adam(params=self.__model.parameters(), lr=lr)
        #
        #     # switch the model status to train-mode
        #     self.__model.train()
        #
        #     print("start training: |", end="")
        #     # train
        #     for i, (samples, labels, _) in enumerate(self.__train_dataloader):
        #
        #         print("-", end="")
        #
        #         samples             = samples.to(self.__device, dtype=torch.float32)
        #         labels              = labels.to(self.__device, dtype=torch.float32)
        #
        #         test = samples.float()
        #
        #         # forward
        #         preds: torch.Tensor = self.__model(samples.float())
        #         train_loss          = self.__criterion(preds, labels.float())
        #
        #         # backward
        #         optimizer.zero_grad()
        #         train_loss.backward()
        #         optimizer.step()
        #
        #     print("|", end="")
        #     print('\n Epoch [%d/%d]  train_loss:%.8f' % (
        #         (epoch + 1), epochs, train_loss.item()))

        for epoch in range(epochs):
            # update the lr
            lr = lr_update_by_epochs_action(lr, epoch)
            optimizer = optim.Adam(params=self.__model.parameters(), lr=lr)

            # switch the model status to train-mode
            self.__model.train()

            with tqdm(total=len(self.__train_dataloader)) as t:
                for i, (samples_batch, labels_batch, _) in enumerate(self.__train_dataloader):
                    samples             = samples_batch.to(self.__device, dtype=torch.float32)
                    labels              = labels_batch.to(self.__device, dtype=torch.float32)

                    # forward
                    preds: torch.Tensor = self.__model(samples.float())
                    train_loss          = self.__criterion(preds, labels.float())

                    # backward
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()

                    t.set_postfix(loss='{:05.9f}'.format(train_loss.item()), lr='{:05.6f}'.format(optimizer.param_groups[0]['lr']))
                    t.update(1)

                print('\n Epoch [%d/%d]  train_loss:%.8f' % (
                    (epoch + 1), epochs, train_loss.item()))

        torch.save(self.__model.state_dict(), self.__name + '.pth')

        return
