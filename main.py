from torch import nn, optim

from models import mcldnn, cnn2

from trainer import train
from dataloaders import rml2016a

from torch.utils.data import DataLoader

import numpy as np

import torch


if __name__ == "__main__":

    seed = 24601
    torch.manual_seed(seed)
    np.random.seed(seed)

    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model: nn.Module = cnn2.CNN2(num_classes=11)

    loss_fn = nn.CrossEntropyLoss()
    optimizer: optim.Optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

    rml2016a_dataset = rml2016a.RML2016aDataset()
    lengths = [int(0.8 * len(rml2016a_dataset)), int(0.2 * len(rml2016a_dataset))]
    train_subset, valid_subset = torch.utils.data.random_split(rml2016a_dataset, lengths)

    train_dataloader = DataLoader(dataset=train_subset, batch_size=512, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_subset, batch_size=512, shuffle=False)

    train.train_and_evaluate(
        model, train_dataloader, valid_dataloader,
        optimizer, loss_fn, device, 200, "CNN2")

