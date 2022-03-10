import torch
import torch.nn as nn
import torch.nn.functional as F


class PairLoss(nn.Module):
    def __init__(self, **kwargs):
        super(PairLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, features, labels):
        device = (torch.device('cuda')
                  if features.is_cuda else torch.device('cpu'))

        batch_size, topk, dim = features.shape

        features = features.reshape((batch_size * topk, dim)).squeeze()
        anchor = labels[:, 0].unsqueeze(1).repeat(1, topk)
        new_labels = (anchor == labels).long()
        new_labels = new_labels.reshape(
            (batch_size * topk, 1)).squeeze().float()
        cls_loss = self.loss(features, new_labels)

        total_loss = cls_loss

        pred = features >= 0.5
        correct = pred.eq(new_labels).sum().item() / float(
            batch_size * topk) * 100.0

        return total_loss
