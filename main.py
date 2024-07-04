from torch import nn, optim

from models import mcldnn
from trainer import train, trainer
from dataloaders import rml2016a

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

import torch

import numpy as np


def train_model(model: nn.Module,
                train_dataloader: DataLoader,
                valid_dataloader: DataLoader,
                epochs: int,
                device: torch.device):

    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):

        model.train()

        if epoch % 15 == 1:
            optimizer = optim.Adam(model.parameters(), lr=lr/10)
            lr /= 10

        train_pred_labels = []
        train_true_labels = []

        validation_pred_labels = []
        validation_true_labels = []

        # train
        for i, (samples, labels, _) in enumerate(train_dataloader):

            samples = samples.to(device, dtype=torch.float32)
            labels = labels.to(device)

            # forward
            preds = model(samples.float())
            train_loss = criterion(preds, labels.float())

            # backward
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            for pred in preds:
                pred_label = np.argmax(pred.cpu().detach().numpy(), axis=0)
                train_pred_labels.append(pred_label)

            for label in labels:
                train_true_labels.append(label.cpu().detach().numpy())

            del preds

        model.eval()
        # validate
        for i, (samples, labels) in enumerate(valid_dataloader):

            samples = samples.to(device, dtype=torch.float32)
            labels = labels.to(device)

            preds = model(samples.float())

            for pred in preds:
                pred_label = np.argmax(pred.cpu().detach().numpy(), axis=0)
                validation_pred_labels.append(pred_label)

            for label in labels:
                validation_true_labels.append(label.cpu().detach().numpy())

            del preds
        model.train()

        accuracy_on_train = accuracy_score(train_true_labels, train_pred_labels)
        accuracy_on_validation = accuracy_score(validation_true_labels, validation_pred_labels)

        print('\nEpoch: [%d/%d]  Train_loss: %.8f  Train_acc: %.3f  Val_acc: %.3f ' % (
            (epoch + 1), epochs, train_loss.item(), accuracy_on_train, accuracy_on_validation))

    return model

if __name__ == "__main__":

    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model: nn.Module = mcldnn.mcldnn(num_classes=11)
    loss_fn = nn.CrossEntropyLoss()
    optimizer: optim.Optimizer = optim.Adam(params=model.parameters(), lr=0e-3)

    rml2016a_dataset = rml2016a.RML2016aDataset()
    train_dataloader = DataLoader(dataset=rml2016a_dataset, batch_size=64)

    # train.train(model, train_dataloader, optimizer, loss_fn, device, 100)

    # Trainer_expr1: trainer.Trainer = trainer.Trainer(name="mcldnn",
    #                                                  model=model,
    #                                                  train_dataloader=train_dataloader,
    #                                                  valid_dataloader=None,
    #                                                  criterion=loss_fn,
    #                                                  device=device)
    #
    # Trainer_expr1.train(100, 1e-3, lambda lr, epochs: lr)

    train_model(model, train_dataloader, None, 100, device)
