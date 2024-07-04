from torch import nn, optim

from models import mcldnn
from trainer import train, trainer
from dataloaders import rml2016a

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

import torch

import numpy as np


if __name__ == "__main__":

    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model: nn.Module = mcldnn.mcldnn(num_classes=11)
    loss_fn = nn.CrossEntropyLoss()
    optimizer: optim.Optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

    rml2016a_dataset = rml2016a.RML2016aDataset()
    train_dataloader = DataLoader(dataset=rml2016a_dataset, batch_size=512, shuffle=True)

    train.train(model, train_dataloader, optimizer, loss_fn, device, 100)

    # Trainer_expr1: trainer.Trainer = trainer.Trainer(name="mcldnn",
    #                                                  model=model,
    #                                                  train_dataloader=train_dataloader,
    #                                                  valid_dataloader=None,
    #                                                  criterion=loss_fn,
    #                                                  device=device)
    #
    # Trainer_expr1.train(100, 1e-3, lambda lr, epochs: lr)

    # train_model(model, train_dataloader, None, 100, device)
