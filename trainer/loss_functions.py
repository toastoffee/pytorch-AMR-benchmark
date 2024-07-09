import torch
import torch.nn as nn
import torch.nn.functional as F


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