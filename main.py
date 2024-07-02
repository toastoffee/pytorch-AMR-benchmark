from torch import nn, optim

from models import mcldnn
from trainer import train
from dataloaders import rml2016a

from torch.utils.data import DataLoader

import torch

if __name__ == "__main__":

    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model: nn.Module = mcldnn.mcldnn(num_classes=11)
    loss_fn = nn.MSELoss()
    optimizer: optim.Optimizer = optim.Adam(params=model.parameters(), lr=5e-3)

    rml2016a_dataset = rml2016a.RML2016aDataset()
    train_dataloader = DataLoader(dataset=rml2016a_dataset, batch_size=64)

    train.train(model, train_dataloader, optimizer, loss_fn, device, 100)
