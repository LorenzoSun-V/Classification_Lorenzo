import torch
import torch.nn as nn


class ClassificationLosses(object):
    def __init__(self, weight=None, reduction='mean', cuda=True):
        self.weight = weight
        self.reduction = reduction
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        criterion = nn.CrossEntropyLoss(weight=self.weight, reduction=self.reduction)
        if self.cuda:
            criterion = criterion.cuda()
        loss = criterion(logit, target.long())
        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=None):
        criterion = nn.CrossEntropyLoss(weight=self.weight, reduction='none')
        if self.cuda:
            criterion = criterion.cuda()
        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise RuntimeError("incorrect reduction!")


if __name__ == "__main__":
    import numpy as np
    loss = ClassificationLosses(cuda=True)
    a = torch.from_numpy(np.array([1, 2, 4, 5])).cuda()
    b = torch.from_numpy(np.array([[0.01, 0.02, 0.01, 0.03, 0.05, 0.05, 0.02],
                                         [0.01, 0.02, 0.01, 0.03, 0.05, 0.05, 0.02],
                                         [0.01, 0.02, 0.01, 0.03, 0.05, 0.05, 0.02],
                                         [0.01, 0.02, 0.01, 0.03, 0.05, 0.05, 0.02]])).cuda()
    # print(loss.CrossEntropyLoss(b, a).item())
    print(loss.FocalLoss(b, a, gamma=2, alpha=0.75).item())
    # print(loss.FocalLoss(b, a, gamma=2, alpha=0.5).item())