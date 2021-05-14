#%%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# %%
# Implementation of a custom dataset

class WineDataset(Dataset):
    
    def __init__(self):
        
        # Data loading
        xy = np.loadtxt('/home/guillaume/Projects/pytorch-learning/data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        
        # Splitting the dataset in separate X, y arrays
        self.X, self.y = torch.from_numpy(xy[:, 1:]), torch.from_numpy(xy[:, [0]].astype(np.int64)) - 1 # using 0 in array to have shape (178, 1)
        
        # Dataset attributes
        self.n_samples, self.n_features = self.X.shape
        
    def __getitem__(self, index):
        # Function allowing user to index in dataset, returns a tuple -> (features, label)
        return self.X[index], self.y[index]
    
    def __len__(self):
        # Function allowing user to use len(dataset) to have n_samples
        return self.n_samples
    
# %%
# Creating the dataset object
dataset = WineDataset()

# Accessing data
first_data = dataset[0]
features, label = first_data
print(features, label)

# %%
# Using a DataLoader to create batches
dl = DataLoader(dataset=dataset, batch_size=42, shuffle=True)

# Accessing the data in dataloader
diter = iter(dl)
data = diter.next()
features, labels = data
print(data)

#%%
# Creating a simple example model
model = nn.Linear(in_features=dataset.n_features, out_features=3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
# %%
# Training loop example with dataloader

# Hyper parameters
num_epochs = 20
total_samples = len(dataset)
n_iters = math.ceil(total_samples / dl.batch_size)

for epoch in range(num_epochs):
    
    # Since we use a dataloader, the training will have steps in each epoch
    for i, (inputs, label) in enumerate(dl):
        
        # Forward pass
        y_predicted = model(inputs)
        
        # Calculating loss
        loss = criterion(y_predicted, label.flatten())
        
        # Calculating gradients
        loss.backward()
        
        # Updating weights
        optimizer.step()
        
        # Emptying gradients
        optimizer.zero_grad()
        
        print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_iters}, loss {loss.item():.4f}')
        