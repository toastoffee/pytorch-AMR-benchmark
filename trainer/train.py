import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm

from train_utils import UpdatingAverage
from sklearn.metrics import accuracy_score


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

    # device adaptation
    model.to(device)
    loss_fn.to(device)

    # start training and use tqdm as the progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (samples_batch, labels_batch) in enumerate(dataloader):

            # convert to torch variables
            samples_batch, labels_batch = samples_batch.to(device), labels_batch.to(device)
            samples_batch, labels_batch = Variable(samples_batch), Variable(labels_batch)

            # set the optimizer grad
            optimizer.zero_grad()

            # compute the network output
            output_batch = model(samples_batch)

            # update weight by loss function
            loss = loss_fn(output_batch, labels_batch)
            loss.backward()
            optimizer.step()

            # update the average loss and accuracy
            loss_avg.update(loss.data, dataloader.batch_size)

            _, predicts_batch = output_batch.max(1)
            accuracy_per_batch = accuracy_score(labels_batch, predicts_batch)
            acc_avg.update(accuracy_per_batch, dataloader.batch_size)

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()), lr='{:05.6f}'.format(optimizer.param_groups[0]['lr']))
            t.update()

        print("- Train accuracy: {acc: .4f}, training loss: {loss: .4f}".format(acc=acc_avg(), loss=loss_avg()))
        return acc_avg, loss_avg
