import torch
import torch.nn as nn

class IrisModel(nn.Module):
    
    def __init__(self, n_features, n_classes, hidden_size=15, lr=0.01):
        super(IrisModel, self).__init__()
        
        # Model paramters
        self.lr = lr
        
        # Model layers
        
        # 1st layer
        self.linear1 = nn.Linear(n_features, hidden_size)
        self.relu = nn.ReLU()
        
        # 2nd layer
        self.linear2 = nn.Linear(hidden_size, n_classes)
        
        
        
    def forward(self, samples):
        ouput = self.linear1(samples)
        output = self.relu(ouput)
        return self.linear2(output)
    
    
    def accuracy(self, y_true, y_predicted):
        
        return torch.sum(y_true == y_predicted) / len(y_true)
