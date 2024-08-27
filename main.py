from torch import nn, optim

from models import mcldnn, cnn2, petcgdnn

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

    # teacher
    # weight_path: str = "./weights/PETCGDNN.pth"
    # teacher_model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
    teacher_model: nn.Module = mcldnn.mcldnn(num_classes=11)
    weight_path: str = "./weights/MCLDNN.pth"
    teacher_model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))

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
    student_model: nn.Module = cnn2.CNN2(num_classes=11)
    student_encoder: nn.Module = cnn2.CNN2_with_projector(num_classes=11, projector_output_dim=128)
    student_encoder: nn.Module = mcldnn.mcldnn_encoder(num_classes=11)
    optimizer: optim.Optimizer = optim.Adam(params=student_model.parameters(), lr=1e-3, weight_decay=0.005)

    # train.train_and_evaluate_dist(student_model, teacher_model, optimizer, train_dataloader, valid_dataloader,
    #                               200, device, "cnn2_mcldnn_dist_mod")

    # train.train_and_evaluate(
    #     model, train_dataloader, valid_dataloader,
    #     optimizer, loss_fn, device, 200, "petcgdnn")

    # sim-KD
    # train.train_and_evaluate_sim_kd_encoder(student_encoder, teacher_encoder, teacher_classifier,  optimizer,
    #                                         train_dataloader, valid_dataloader, 200,  device,
    #                                         "cnn2_mcldnn_128_simKd")

    loss_fn = nn.CrossEntropyLoss()

    resnet2: nn.Module = resnet1d.resnet2(num_class=11)
    resnet4: nn.Module = resnet1d.resnet4(num_class=11)
    resnet10: nn.Module = resnet1d.resnet10(num_class=11)
    resnet18: nn.Module = resnet1d.resnet18(num_class=11)
    resnet34: nn.Module = resnet1d.resnet34(num_class=11)
    resnet50: nn.Module = resnet1d.resnet50(num_class=11)
    resnet101: nn.Module = resnet1d.resnet101(num_class=11)
    resnet152: nn.Module = resnet1d.resnet152(num_class=11)

    optimizer2: optim.Optimizer = optim.Adam(params=resnet2.parameters(), lr=1e-3, weight_decay=0.005)
    optimizer4: optim.Optimizer = optim.Adam(params=resnet4.parameters(), lr=1e-3, weight_decay=0.005)
    optimizer10: optim.Optimizer = optim.Adam(params=resnet10.parameters(), lr=1e-3, weight_decay=0.005)
    optimizer18: optim.Optimizer = optim.Adam(params=resnet18.parameters(), lr=1e-3, weight_decay=0.005)
    optimizer34: optim.Optimizer = optim.Adam(params=resnet34.parameters(), lr=1e-3, weight_decay=0.005)
    optimizer50: optim.Optimizer = optim.Adam(params=resnet50.parameters(), lr=1e-3, weight_decay=0.005)
    optimizer101: optim.Optimizer = optim.Adam(params=resnet101.parameters(), lr=1e-3, weight_decay=0.005)
    optimizer152: optim.Optimizer = optim.Adam(params=resnet152.parameters(), lr=1e-3, weight_decay=0.005)

    # initial dataset
    rml2016a_dataset = rml2016a.RML2016aDataset()
    lengths = [int(0.6 * len(rml2016a_dataset)), int(0.4 * len(rml2016a_dataset))]
    train_subset, valid_subset = torch.utils.data.random_split(rml2016a_dataset, lengths)
    train_dataloader = DataLoader(dataset=train_subset, batch_size=512, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_subset, batch_size=512, shuffle=False)


    # 1. train 6-resnets baseline model
    # train.train_and_evaluate(
    #     resnet2, train_dataloader, valid_dataloader,
    #     optimizer2, loss_fn, device, 50, "resnet2-baseline")

    # train.train_and_evaluate(
    #     resnet4, train_dataloader, valid_dataloader,
    #     optimizer4, loss_fn, device, 200, "resnet4-baseline")
    #
    # train.train_and_evaluate(
    #     resnet10, train_dataloader, valid_dataloader,
    #     optimizer10, loss_fn, device, 200, "resnet10-baseline")
    #
    # train.train_and_evaluate(
    #     resnet18, train_dataloader, valid_dataloader,
    #     optimizer18, loss_fn, device, 200, "resnet18-baseline")
    #
    # train.train_and_evaluate(
    #     resnet34, train_dataloader, valid_dataloader,
    #     optimizer34, loss_fn, device, 10, "resnet34-baseline")
    #
    # train.train_and_evaluate(
    #     resnet50, train_dataloader, valid_dataloader,
    #     optimizer50, loss_fn, device, 200, "resnet50-baseline")
    #
    # train.train_and_evaluate(
    #     resnet101, train_dataloader, valid_dataloader,
    #     optimizer101, loss_fn, device, 200, "resnet101-baseline")
    #
    # train.train_and_evaluate(
    #     resnet152, train_dataloader, valid_dataloader,
    #     optimizer152, loss_fn, device, 200, "resnet152-baseline")

    # 2. traditional-KD
    # 4 teacher model
    # resnet34: nn.Module = resnet1d.resnet34(num_class=11)
    # resnet50: nn.Module = resnet1d.resnet50(num_class=11)
    # resnet101: nn.Module = resnet1d.resnet101(num_class=11)
    # resnet152: nn.Module = resnet1d.resnet152(num_class=11)

    cnn2_1: nn.Module = cnn2.CNN2(num_classes=11)
    cnn2_2: nn.Module = cnn2.CNN2(num_classes=11)
    cnn2_3: nn.Module = cnn2.CNN2(num_classes=11)
    cnn2_4: nn.Module = cnn2.CNN2(num_classes=11)
    resnet18_1: nn.Module = resnet1d.resnet18(num_class=11)
    resnet18_2: nn.Module = resnet1d.resnet18(num_class=11)
    resnet18_3: nn.Module = resnet1d.resnet18(num_class=11)
    resnet18_4: nn.Module = resnet1d.resnet18(num_class=11)

    optimizer_cnn2_1: optim.Optimizer = optim.Adam(params=cnn2_1.parameters(), lr=1e-3, weight_decay=0.005)
    optimizer_cnn2_2: optim.Optimizer = optim.Adam(params=cnn2_2.parameters(), lr=1e-3, weight_decay=0.005)
    optimizer_cnn2_3: optim.Optimizer = optim.Adam(params=cnn2_3.parameters(), lr=1e-3, weight_decay=0.005)
    optimizer_cnn2_4: optim.Optimizer = optim.Adam(params=cnn2_4.parameters(), lr=1e-3, weight_decay=0.005)
    optimizer18_1: optim.Optimizer = optim.Adam(params=resnet18_1.parameters(), lr=1e-3, weight_decay=0.005)
    optimizer18_2: optim.Optimizer = optim.Adam(params=resnet18_2.parameters(), lr=1e-3, weight_decay=0.005)
    optimizer18_3: optim.Optimizer = optim.Adam(params=resnet18_3.parameters(), lr=1e-3, weight_decay=0.005)
    optimizer18_4: optim.Optimizer = optim.Adam(params=resnet18_4.parameters(), lr=1e-3, weight_decay=0.005)

    # resnet34_weight_path: str = "./weights/resnet34-baseline.pth"
    # resnet34.load_state_dict(torch.load(resnet34_weight_path, map_location=torch.device(device)))
    #
    # resnet50_weight_path: str = "./weights/resnet50-baseline.pth"
    # resnet50.load_state_dict(torch.load(resnet50_weight_path, map_location=torch.device(device)))
    #
    # resnet101_weight_path: str = "./weights/resnet101-baseline.pth"
    # resnet101.load_state_dict(torch.load(resnet101_weight_path, map_location=torch.device(device)))
    #
    # resnet152_weight_path: str = "./weights/resnet152-baseline.pth"
    # resnet152.load_state_dict(torch.load(resnet152_weight_path, map_location=torch.device(device)))

    # 2 student model : resnet18 and cnn2

    train.train_and_evaluate_kd(student_model, teacher_model, optimizer, train_dataloader, valid_dataloader,
                                200, device,0.1, 4,  "cnn2_petcgdnn_vanilla")

    train.train_and_evaluate_kd(model=cnn2_1, teacher_model=resnet34, optimizer=optimizer_cnn2_1, train_dataloader=train_dataloader, val_dataloader=valid_dataloader,
                                epochs=50, device=device, alpha=0.9, temperature=4, model_name="cnn2_res34_vanilla")
    train.train_and_evaluate_kd(cnn2_2, resnet50, optimizer_cnn2_2, train_dataloader, valid_dataloader,
                                50, device, 0.9, 4, "cnn2_res50_vanilla")
    train.train_and_evaluate_kd(cnn2_3, resnet101, optimizer_cnn2_3, train_dataloader, valid_dataloader,
                                50, device, 0.9, 4, "cnn2_res101_vanilla")
    train.train_and_evaluate_kd(cnn2_4, resnet152, optimizer_cnn2_4, train_dataloader, valid_dataloader,
                                50, device, 0.9, 4, "cnn2_res152_vanilla")

    train.train_and_evaluate_kd(resnet18_1, resnet34, optimizer18_1, train_dataloader, valid_dataloader,
                                50, device, 0.9, 4, "res18_res34_vanilla")
    train.train_and_evaluate_kd(resnet18_2, resnet50, optimizer18_2, train_dataloader, valid_dataloader,
                                50, device, 0.9, 4, "res18_res50_vanilla")
    train.train_and_evaluate_kd(resnet18_3, resnet101, optimizer18_3, train_dataloader, valid_dataloader,
                                50, device, 0.9, 4, "res18_res101_vanilla")
    train.train_and_evaluate_kd(resnet18_4, resnet152, optimizer18_4, train_dataloader, valid_dataloader,
                                50, device, 0.9, 4, "res18_res152_vanilla")
    # 3. dist-KD

