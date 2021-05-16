import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    
    
    def __init__(self, lr=0.0001):
        super(CNN, self).__init__()
        
        self.lr = lr
        
        # Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=69)  # Nice
        self.fc3 = nn.Linear(in_features=69, out_features=10)
        

    def forward(self, samples):
        
        # 1st conv layer
        out = F.relu(self.conv1(samples))
        out = self.pool(out)
        
        # 2nd conv layer
        out = F.relu(self.conv2(out))
        out = self.pool(out)
        
        # Flattening the tensor
        out = out.reshape(-1, 16*5*5)
        
        out = self.fc1(out)
        out = F.relu(out)
        
        out = self.fc2(out)
        out = F.relu(out)
        
        return self.fc3(out)
