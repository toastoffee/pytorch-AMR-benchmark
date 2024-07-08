import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from tqdm import tqdm

# from train_utils import UpdatingAverage
from sklearn.metrics import accuracy_score


class UpdatingAverage:
    """
    record the float value and return the average of records
    """

    def __init__(self):
        self.steps: int   = 0
        self.sum:   float = 0

    def update(self, val):
        self.sum   += val
        self.steps += 1

    def __call__(self, *args, **kwargs):
        return self.sum / float(self.steps)


def train_and_evaluate(model:        nn.Module,
                       train_dataloader:   DataLoader,
                       val_dataloader: DataLoader,
                       optimizer:    optim.Optimizer,
                       loss_fn:      nn.Module,
                       device:       torch.device,
                       epochs:        int,
                       model_name:   str):
    # device adaptation
    model.to(device)
    loss_fn.to(device)

    # learning rate schedulers
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    for epoch in range(epochs):

        if epoch > 0:
            scheduler.step()

        # train and evaluate the model
        train_one_epoch(model, train_dataloader, optimizer, loss_fn, device)
        evaluate(model, val_dataloader, loss_fn, device)

    torch.save(model.state_dict(),  model_name + '.pth')


# regular train function
def train(model:        nn.Module,
          dataloader:   DataLoader,
          optimizer:    optim.Optimizer,
          loss_fn:      nn.Module,
          device:       torch.device,
          epochs:        int):

    # device adaptation
    model.to(device)
    loss_fn.to(device)

    # learning rate schedulers
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    for epoch in range(epochs):

        if epoch > 0:
            scheduler.step()

        # train the model
        train_one_epoch(model, dataloader, optimizer, loss_fn, device)


# regular train with only one epoch
def train_one_epoch(model:      nn.Module,
                    dataloader: DataLoader,
                    optimizer:  optim.Optimizer,
                    loss_fn:    nn.Module,
                    device:     torch.device):

    # set the model to training mode
    model.train()

    # metrics
    loss_avg = UpdatingAverage()
    acc_avg = UpdatingAverage()

    # start training and use tqdm as the progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (samples, labels, snr) in enumerate(dataloader):

            # convert to torch variables
            samples, labels = samples.to(device, dtype=torch.float32), labels.to(device)

            # forward
            preds: torch.Tensor = model(samples)
            loss = loss_fn(preds.float(), labels.long())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the average loss and accuracy
            loss_avg.update(loss.data)

            pred_labels = torch.argmax(preds, dim=1)
            accuracy_per_batch = accuracy_score(pred_labels.cpu(), labels.cpu())
            acc_avg.update(accuracy_per_batch)

            t.set_postfix(loss='{:05.8f}'.format(loss_avg()), lr='{:05.6f}'.format(optimizer.param_groups[0]['lr']))
            t.update(1)

        print("- Train metrics, acc: {acc: .4f}, loss: {loss: .4f}".format(acc=acc_avg(), loss=loss_avg()))
        return acc_avg, loss_avg


# evaluate the mode
def evaluate(model:        nn.Module,
             dataloader:   DataLoader,
             loss_fn:      nn.Module,
             device: torch.device):

    model.eval()

    loss_avg = UpdatingAverage()
    acc_avg = UpdatingAverage()

    for i, (samples, labels, snr) in enumerate(dataloader):
        samples, labels = samples.to(device, dtype=torch.float32), labels.to(device)

        preds: torch.Tensor = model(samples)
        loss = loss_fn(preds.float(), labels.long())

        loss_avg.update(loss.data)

        pred_labels = torch.argmax(preds, dim=1)
        accuracy_per_batch = accuracy_score(pred_labels.cpu(), labels.cpu())
        acc_avg.update(accuracy_per_batch)

    print("- Eval metrics, acc: {acc: .4f}, loss: {loss: .4f}".format(acc=acc_avg(), loss=loss_avg()))
