from torch import nn, optim

from models import mcldnn, cnn2, petcgdnn

from trainer import train, loss_functions
from dataloaders import rml2016a

from torch.utils.data import DataLoader

import numpy as np

import torch


if __name__ == "__main__":

    seed = 24601
    torch.manual_seed(seed)
    np.random.seed(seed)

    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # teacher
    model: nn.Module = petcgdnn.PETCGDNN(num_classes=11)
    # weight_path: str = "./weights/PETCGDNN.pth"
    # teacher_model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
    # teacher_model: nn.Module = mcldnn.mcldnn(num_classes=11)
    # weight_path: str = "./weights/MCLDNN.pth"
    # teacher_model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))

    # teacher_encoder: nn.Module = petcgdnn.PETCGDNN_encoder(num_classes=11)
    # teacher_classifier: nn.Module = petcgdnn.PETCGDNN_classifier(num_classes=11)
    # weight_path: str = "./weights/PETCGDNN.pth"
    # teacher_encoder.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
    # teacher_classifier.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))

    # teacher_encoder: nn.Module = mcldnn.mcldnn_encoder(num_classes=11)
    # teacher_classifier: nn.Module = mcldnn.mcldnn_classifier(num_classes=11)
    # weight_path: str = "./weights/MCLDNN.pth"
    # teacher_encoder.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
    # teacher_classifier.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))

    # student
    # student_model: nn.Module = cnn2.CNN2(num_classes=11)
    # student_encoder: nn.Module = cnn2.CNN2_with_projector(num_classes=11, projector_output_dim=128)
    # student_encoder: nn.Module = mcldnn.mcldnn_encoder(num_classes=11)

    loss_fn = nn.CrossEntropyLoss()

    optimizer: optim.Optimizer = optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=0.005)

    rml2016a_dataset = rml2016a.RML2016aDataset()
    lengths = [int(0.6 * len(rml2016a_dataset)), int(0.4 * len(rml2016a_dataset))]
    train_subset, valid_subset = torch.utils.data.random_split(rml2016a_dataset, lengths)

    train_dataloader = DataLoader(dataset=train_subset, batch_size=512, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_subset, batch_size=512, shuffle=False)

    # train.train_and_evaluate_kd(student_model, teacher_model, optimizer, train_dataloader, valid_dataloader,
    #                             200, device, 0.9, 4, "cnn2_petcgdnn_vanilla")

    # train.train_and_evaluate_dist(student_model, teacher_model, optimizer, train_dataloader, valid_dataloader,
    #                               200, device, "cnn2_mcldnn_dist_mod")

    train.train_and_evaluate(
        model, train_dataloader, valid_dataloader,
        optimizer, loss_fn, device, 200, "petcgdnn")

    # sim-KD
    # train.train_and_evaluate_sim_kd_encoder(student_encoder, teacher_encoder, teacher_classifier,  optimizer,
    #                                         train_dataloader, valid_dataloader, 200,  device,
    #                                         "cnn2_mcldnn_128_simKd")
