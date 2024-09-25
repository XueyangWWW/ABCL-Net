import torch

from torch import nn
from torch.nn.functional import max_pool3d
import numpy as np

# ori_loss
class dice_coef(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        smooth = 1e-6
        size = y_true.shape[-1] // y_pred.shape[-1]
        y_true = max_pool3d(y_true, size, size)
        a = torch.sum(y_true * y_pred)
        b = torch.sum(y_true)
        c = torch.sum(y_pred)
        dice = (2 * a) / (b + c + smooth)
        return torch.mean(dice)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class mix_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):

        return crossentropy()(y_pred, y_true) + 1 - dice_coef()(y_true, y_pred)


class crossentropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_pred, y_true)
        return loss
