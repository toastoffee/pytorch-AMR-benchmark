from torch import nn, optim

from models import mcldnn, cnn2, petcgdnn, fits

from trainer import train, loss_functions
from dataloaders import rml2016a

from torch.utils.data import DataLoader

import numpy as np

import torch

from models import resnet1d

if __name__ == "__main__":
    seed = 24601
    torch.manual_seed(seed)
    np.random.seed(seed)

    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    loss_fn = nn.CrossEntropyLoss()

    # initial dataset
    rml2016a_dataset = rml2016a.RML2016aDataset()
    lengths = [int(0.6 * len(rml2016a_dataset)), int(0.4 * len(rml2016a_dataset))]
    train_subset, valid_subset = torch.utils.data.random_split(rml2016a_dataset, lengths)
    train_dataloader = DataLoader(dataset=train_subset, batch_size=512, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_subset, batch_size=512, shuffle=False)

    # 1. train 6-resnets baseline model
    # resnet2: nn.Module = resnet1d.resnet2(num_class=11)
    # optimizer2: optim.Optimizer = optim.Adam(params=resnet2.parameters(), lr=1e-3, weight_decay=0.005)
    #
    # train.train_and_evaluate(
    #     resnet2, train_dataloader, valid_dataloader,
    #     optimizer2, loss_fn, device, 50, "resnet2-baseline")

    resnet2: nn.Module = fits.FitsWithResnet(128, 0, False, 2, 60)
    optimizer2: optim.Optimizer = optim.Adam(params=resnet2.parameters(), lr=1e-3, weight_decay=0.005)

    train.train_and_evaluate(
        resnet2, train_dataloader, valid_dataloader,
        optimizer2, loss_fn, device, 50, "resnet-fits-baseline")
