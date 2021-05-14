import torch
import torch.nn as nn

class IrisModel(nn.Module):
    
    
    def __init__(self, n_features, n_classes, lr=0.01):
        super(IrisModel, self).__init__()
        
        # Model paramters
        self.lr = lr
        
        # Model layers
        self.linear = nn.Linear(n_features, n_classes)
        
        
    def forward(self, samples):
        return self.linear(samples)
    
    
    def accuracy(self, y_true, y_predicted):
        
        return torch.sum(y_true == y_predicted) / len(y_true)
