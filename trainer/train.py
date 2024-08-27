import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn.functional as F

from tqdm import tqdm

# from train_utils import UpdatingAverage
from sklearn.metrics import accuracy_score, top_k_accuracy_score


def log_info(content: str):
    with open("log.txt", "a") as file:
        file.write(content + '\n')


def loss_kd(outputs:            torch.Tensor,
            labels:             torch.Tensor,
            teacher_outputs:    torch.Tensor,
            alpha:              float,
            T:        int):
    """
    loss function for Knowledge Distillation (KD)
    """

    loss_CE = F.cross_entropy(outputs, labels)
    D_KL = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (T * T)
    KD_loss =  (1. - alpha)*loss_CE + alpha*D_KL

    return KD_loss


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


def cosine_similarity(x, y, eps=1e-8):
    return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + eps)


def pearson_correlation(x, y, eps=1e-8):
    return cosine_similarity(x - x.mean(1).unsqueeze(1), y - y.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


def normalized_cross_entropy(y_s : torch.Tensor, y_t: torch.Tensor):
    y_s = torch.nn.functional.normalize(y_s, dim=1, p=2);
    y_t = torch.nn.functional.normalize(y_t, dim=1, p=2);
    return F.kl_div(y_s, y_t)




class DIST(nn.Module):
    def __init__(self, beta=1., gamma=1.):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_s, y_t):
        assert y_s.ndim in (2, 4)
        if y_s.ndim == 4:
            num_classes = y_s.shape[1]
            y_s = y_s.transpose(1, 3).reshape(-1, num_classes)
            y_t = y_t.transpose(1, 3).reshape(-1, num_classes)
        y_s = y_s.softmax(dim=1)
        y_t = y_t.softmax(dim=1)
        inter_loss = inter_class_relation(y_s, y_t)
        intra_loss = intra_class_relation(y_s, y_t)
        loss = self.beta * inter_loss + self.gamma * intra_loss
        return loss


class DIST_M(nn.Module):
    def __init__(self, beta=1., gamma=1.):
        super(DIST_M, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_s, y_t):
        assert y_s.ndim in (2, 4)
        if y_s.ndim == 4:
            num_classes = y_s.shape[1]
            y_s = y_s.transpose(1, 3).reshape(-1, num_classes)
            y_t = y_t.transpose(1, 3).reshape(-1, num_classes)
        y_s = y_s.softmax(dim=1)
        y_t = y_t.softmax(dim=1)
        inter_loss = normalized_cross_entropy(y_s, y_t)
        intra_loss = normalized_cross_entropy(y_s.transpose(0, 1), y_t.transpose(0, 1))
        loss = self.beta * inter_loss + self.gamma * intra_loss
        return loss


def train_and_evaluate(model:        nn.Module,
                       train_dataloader:   DataLoader,
                       val_dataloader: DataLoader,
                       optimizer:    optim.Optimizer,
                       loss_fn:      nn.Module,
                       device:       torch.device,
                       epochs:        int,
                       model_name:   str):

    log_info("start training: " + model_name)
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
    # scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1, last_epoch=-1)

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
             device: torch.device,
             desc: str = ""):

    model.eval()

    loss_avg = UpdatingAverage()
    acc_avg = UpdatingAverage()
    acc5_avg = UpdatingAverage()

    for i, (samples, labels, snr) in enumerate(dataloader):
        samples, labels = samples.to(device, dtype=torch.float32), labels.to(device)

        preds: torch.Tensor = model(samples)
        loss = loss_fn(preds.float(), labels.long())

        loss_avg.update(loss.data)

        pred_labels = torch.argmax(preds, dim=1)
        accuracy_per_batch = accuracy_score(pred_labels.cpu(), labels.cpu())
        accuracy_top_5 = top_k_accuracy_score(y_true=labels.cpu().detach().numpy(), y_score=preds.cpu().detach().numpy(), k=5)
        acc_avg.update(accuracy_per_batch)
        acc5_avg.update(accuracy_top_5)

    metric_desc = desc + "- Eval metrics, acc: {acc: .4f},top-5:{top5: .4f} loss: {loss: .4f}".format(acc=acc_avg(), top5=acc5_avg(), loss=loss_avg())
    print(desc + "- Eval metrics, acc: {acc: .4f}, loss: {loss: .4f}".format(acc=acc_avg(), loss=loss_avg()))
    log_info(metric_desc)

def train_kd(model:             nn.Module,
             teacher_model:     nn.Module,
             optimizer:         optim.Optimizer,
             dataloader:        DataLoader,
             epochs:             int,
             device:            torch.device,
             alpha:             float,
             temperature:       int):

    """
    KD Train the model on `num_steps` batches
    """
    # set model to training mode
    model.train()
    teacher_model.eval()

    model.to(device)
    teacher_model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1, last_epoch=-1)
    for epoch in range(epochs):

        if epoch > 0:
            scheduler.step()

        # train the model
        train_kd_one_epoch(model, teacher_model, dataloader, optimizer, device, alpha, temperature)


def train_and_evaluate_kd(model:             nn.Module,
                          teacher_model:     nn.Module,
                          optimizer:         optim.Optimizer,
                          train_dataloader:  DataLoader,
                          val_dataloader:    DataLoader,
                          epochs:            int,
                          device:            torch.device,
                          alpha:             float,
                          temperature:       int,
                          model_name:        str):

    """
    KD Train the model on `num_steps` batches
    """
    # set model to training mode
    model.train()
    teacher_model.eval()

    model.to(device)
    teacher_model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1, last_epoch=-1)
    for epoch in range(epochs):

        if epoch > 0:
            scheduler.step()

        # train the model
        train_kd_one_epoch(model, teacher_model, train_dataloader, optimizer, device, alpha, temperature)

        loss_fn = nn.CrossEntropyLoss()
        evaluate(model, val_dataloader, loss_fn, device)

    torch.save(model.state_dict(),  model_name + '.pth')


def train_kd_one_epoch(model:         nn.Module,
                       teacher_model: nn.Module,
                       dataloader:    DataLoader,
                       optimizer:     optim.Optimizer,
                       device:        torch.device,
                       alpha:         float,
                       temperature:   int):

    # set the model to training mode
    model.train()
    teacher_model.eval()

    # metrics
    loss_avg = UpdatingAverage()
    acc_avg = UpdatingAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (samples, labels, snr) in enumerate(dataloader):

            samples, labels = samples.to(device), labels.to(device)
            # convert to torch Variables
            samples, labels = Variable(samples), Variable(labels)

            # compute model output, fetch teacher output, and compute KD loss
            preds:  torch.Tensor = model(samples).to(device)

            # get one batch output from teacher model
            teacher_preds = teacher_model(samples).to(device)
            teacher_preds = Variable(teacher_preds, requires_grad=False)

            loss = loss_kd(preds, labels, teacher_preds, alpha, temperature)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_labels = torch.argmax(preds, dim=1)
            accuracy_per_batch = accuracy_score(pred_labels.cpu(), labels.cpu())
            acc_avg.update(accuracy_per_batch)

            # update the average loss
            loss_avg.update(loss.data)

            t.set_postfix(loss='{:05.8f}'.format(loss_avg()), lr='{:05.8f}'.format(optimizer.param_groups[0]['lr']))
            t.update()

    print("- Train metrics, acc: {acc: .4f}, loss: {loss: .4f}".format(acc=acc_avg(), loss=loss_avg()))
    return acc_avg, loss_avg

# modified version
# def train_dist_one_epoch(model:         nn.Module,
#                        teacher_model: nn.Module,
#                        dataloader:    DataLoader,
#                        optimizer:     optim.Optimizer,
#                        device:        torch.device):
#
#     # set the model to training mode
#     model.train()
#     teacher_model.eval()
#
#     # metrics
#     loss_avg = UpdatingAverage()
#     acc_avg = UpdatingAverage()
#
#     # Use tqdm for progress bar
#     with tqdm(total=len(dataloader)) as t:
#         for i, (samples, labels, snr) in enumerate(dataloader):
#
#             samples, labels = samples.to(device), labels.to(device)
#             # convert to torch Variables
#             samples, labels = Variable(samples), Variable(labels)
#
#             # compute model output, fetch teacher output, and compute KD loss
#             preds:  torch.Tensor = model(samples).to(device)
#
#             # get one batch output from teacher model
#             teacher_preds = teacher_model(samples).to(device)
#             teacher_preds = Variable(teacher_preds, requires_grad=False)
#
#             loss_ce = F.cross_entropy(preds, labels)
#             dist = DIST_M()
#             dist_loss = dist.forward(preds, teacher_preds)
#
#             loss = 0.33 * loss_ce + 0.66 * dist_loss
#
#             # clear previous gradients, compute gradients of all variables wrt loss
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             pred_labels = torch.argmax(preds, dim=1)
#             accuracy_per_batch = accuracy_score(pred_labels.cpu(), labels.cpu())
#             acc_avg.update(accuracy_per_batch)
#
#             # update the average loss
#             loss_avg.update(loss.data)
#
#             t.set_postfix(loss='{:05.8f}'.format(loss_avg()), lr='{:05.8f}'.format(optimizer.param_groups[0]['lr']))
#             t.update()
#
#     print("- Train metrics, acc: {acc: .4f}, loss: {loss: .4f}".format(acc=acc_avg(), loss=loss_avg()))
#     return acc_avg, loss_avg

def train_dist_one_epoch(model:         nn.Module,
                       teacher_model: nn.Module,
                       dataloader:    DataLoader,
                       optimizer:     optim.Optimizer,
                       device:        torch.device):

    # set the model to training mode
    model.train()
    teacher_model.eval()

    # metrics
    loss_avg = UpdatingAverage()
    acc_avg = UpdatingAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (samples, labels, snr) in enumerate(dataloader):

            samples, labels = samples.to(device), labels.to(device)
            # convert to torch Variables
            samples, labels = Variable(samples), Variable(labels)

            # compute model output, fetch teacher output, and compute KD loss
            preds:  torch.Tensor = model(samples).to(device)

            # get one batch output from teacher model
            teacher_preds = teacher_model(samples).to(device)
            teacher_preds = Variable(teacher_preds, requires_grad=False)

            loss_ce = F.cross_entropy(preds, labels)
            dist = DIST()
            dist_loss = dist.forward(preds, teacher_preds)

            loss = 0.5 * loss_ce + 0.5 * dist_loss

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_labels = torch.argmax(preds, dim=1)
            accuracy_per_batch = accuracy_score(pred_labels.cpu(), labels.cpu())
            acc_avg.update(accuracy_per_batch)

            # update the average loss
            loss_avg.update(loss.data)

            t.set_postfix(loss='{:05.8f}'.format(loss_avg()), lr='{:05.8f}'.format(optimizer.param_groups[0]['lr']))
            t.update()

    print("- Train metrics, acc: {acc: .4f}, loss: {loss: .4f}".format(acc=acc_avg(), loss=loss_avg()))
    return acc_avg, loss_avg

def train_and_evaluate_dist(model:             nn.Module,
                          teacher_model:     nn.Module,
                          optimizer:         optim.Optimizer,
                          train_dataloader:  DataLoader,
                          val_dataloader:    DataLoader,
                          epochs:            int,
                          device:            torch.device,
                          model_name:        str):

    """
    KD Train the model on `num_steps` batches
    """
    # set model to training mode
    model.train()
    teacher_model.eval()

    model.to(device)
    teacher_model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1, last_epoch=-1)
    for epoch in range(epochs):

        if epoch > 0:
            scheduler.step()

        # train the model
        train_dist_one_epoch(model, teacher_model, train_dataloader, optimizer, device)

        loss_fn = nn.CrossEntropyLoss()
        evaluate(model, val_dataloader, loss_fn, device)

    torch.save(model.state_dict(),  model_name + '.pth')


def train_sim_one_epoch(student_encoder: nn.Module,
                        teacher_encoder: nn.Module,
                        dataloader:      DataLoader,
                        optimizer:       optim.Optimizer,
                        device:          torch.device):

    # set the model to training mode
    student_encoder.train()
    teacher_encoder.eval()

    # metrics
    loss_avg = UpdatingAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (samples, labels, snr) in enumerate(dataloader):

            samples = samples.to(device)
            # convert to torch Variables
            samples = Variable(samples)

            # compute model output, fetch teacher output, and compute KD loss
            s_encodes:  torch.Tensor = student_encoder(samples).to(device)

            # get one batch output from teacher model
            t_encodes = teacher_encoder(samples).to(device)
            t_encodes = Variable(t_encodes, requires_grad=False)

            loss = F.mse_loss(s_encodes, t_encodes)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the average loss
            loss_avg.update(loss.data)

            t.set_postfix(loss='{:05.8f}'.format(loss_avg()), lr='{:05.8f}'.format(optimizer.param_groups[0]['lr']))
            t.update()

    print("- sim-KD encoder Train metrics, loss: {loss: .4f}".format(loss=loss_avg()))
    return loss_avg


def train_sim_kd_encoder(student_encoder:   nn.Module,
                         teacher_encoder:   nn.Module,
                         optimizer:         optim.Optimizer,
                         dataloader:        DataLoader,
                         epochs:            int,
                         device:            torch.device,
                         model_name:        str):

    """
    KD Train the model on `num_steps` batches
    """
    # set model to training mode
    student_encoder.train()
    teacher_encoder.eval()

    student_encoder.to(device)
    teacher_encoder.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1, last_epoch=-1)
    for epoch in range(epochs):

        if epoch > 0:
            scheduler.step()

        # train the model
        train_sim_one_epoch(student_encoder, teacher_encoder, dataloader, optimizer, device)

    torch.save(student_encoder.state_dict(),  model_name + '.pth')


def train_and_evaluate_sim_kd_encoder(student_encoder:    nn.Module,
                                      teacher_encoder:    nn.Module,
                                      teacher_classifier: nn.Module,
                                      optimizer:          optim.Optimizer,
                                      train_dataloader:   DataLoader,
                                      val_dataloader:     DataLoader,
                                      epochs:             int,
                                      device:             torch.device,
                                      model_name:         str):

    """
    KD Train the model on `num_steps` batches
    """
    # set model to training mode
    student_encoder.train()
    teacher_encoder.eval()
    teacher_classifier.eval()

    student_encoder.to(device)
    teacher_encoder.to(device)
    teacher_classifier.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1, last_epoch=-1)
    for epoch in range(epochs):

        if epoch > 0:
            scheduler.step()

        # train the model
        train_sim_one_epoch(student_encoder, teacher_encoder, train_dataloader, optimizer, device)

        loss_fn = nn.CrossEntropyLoss()

        evaluate(nn.Sequential(teacher_encoder, teacher_classifier), val_dataloader, loss_fn, device, "teacher ")
        evaluate(nn.Sequential(student_encoder, teacher_classifier), val_dataloader, loss_fn, device, "student ")

    torch.save(student_encoder.state_dict(),  model_name + '.pth')