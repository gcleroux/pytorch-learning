import torch
import torch.nn as nn


class BCModel(nn.Module):

    def __init__(self, n_features, lr=0.00001):
        super(BCModel, self).__init__()

        # Model paramters
        self.lr = lr

        # Model layers
        self.linear = nn.Linear(in_features=n_features, out_features=1)

    def forward(self, samples):
        return self.linear(samples)

    def accuracy(self, y_predicted, y_true):
        
        y_predicted = torch.tensor(
            [1 if y > 0 else 0 for y in y_predicted])

        return torch.sum(y_predicted == y_true.flatten()) / len(y_true)
