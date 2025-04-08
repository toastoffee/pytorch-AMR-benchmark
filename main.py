from torch import nn, optim

from models import mcldnn, cnn2, petcgdnn, fits

from trainer import train, loss_functions
from dataloaders import rml2016a

from torch.utils.data import DataLoader

import numpy as np

import torch

from models import resnet1d
from models.modern_tcn import modernTCN

from models import gru
from models.cnn2 import CNN2
from models.mcldnn import mcldnn
from models.petcgdnn import PETCGDNN

if __name__ == "__main__":
    seed = 24601
    torch.manual_seed(seed)
    np.random.seed(seed)

    device: torch.device = None
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    rml2016a = rml2016a.RML2016aDataset()
    lengths = [int(0.6 * len(rml2016a)), int(0.4 * len(rml2016a))]
    train_subset, valid_subset = torch.utils.data.random_split(rml2016a, lengths)
    train_dataloader = DataLoader(dataset=train_subset, batch_size=512, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_subset, batch_size=512, shuffle=False)
    #
    # mse_loss = nn.MSELoss()
    #
    # net = fits.MyFits(96, 32, False, 2, 48)
    # optimizer: optim.Optimizer = optim.Adam(params=net.parameters(), lr=1e-3, weight_decay=0.005)
    #
    # train.train_mse(
    #     net, train_dataloader,
    #     optimizer, mse_loss, device, 1)
    #
    loss_fn = nn.CrossEntropyLoss()


    # 1. train 6-resnets baseline model
    net: nn.Module = PETCGDNN(num_classes=11)
    optimizer: optim.Optimizer = optim.Adam(params=net.parameters(), lr=1e-3, weight_decay=0.005)


    train.train_and_evaluate(
        net, train_dataloader, valid_dataloader,
        optimizer, loss_fn, device, 50, "cnn1-baseline")
