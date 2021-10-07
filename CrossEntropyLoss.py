import torch
from torch import autograd
from torch import nn


class CrossEntropyLoss(nn.Module):
    log_softmax = nn.LogSoftmax()

    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = autograd.Variable(torch.FloatTensor(class_weights).cuda())

    def forward(self, logits, target):
        log_probabilities = self.log_softmax(logits)
        return -self.class_weights.index_select(0, target) * log_probabilities.index_select(-1, target).diag()
