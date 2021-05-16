import torch.nn as nn

class NeuralNet(nn.Module):
    
    
    def __init__(self, input_size, hidden_size, output_size, lr=0.0001):
        
        super(NeuralNet, self).__init__()
        
        self.lr = lr
        
        # Layers
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)
        
    
    def forward(self, samples):
        
        out = self.l1(samples)
        out = self.relu(out)
        return self.l2(out)
    