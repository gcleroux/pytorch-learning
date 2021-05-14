#%%
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision import transforms


# %%
# Trying out the torchvision datasets with dataloader
ds = torchvision.datasets.CIFAR10(root='./data', transform=torchvision.transforms.ToTensor(), download=True)

dls = DataLoader(ds, batch_size=50, shuffle=True, num_workers=2)

# Accessing elements of the batch
diter = iter(dls)
data = diter.next()
images, labels = data
print(images, labels)

# %%
# Applying a custom transform class to the wine dataset

class WineDataset(Dataset):

    def __init__(self, transform=None):

        # Data loading
        xy = np.loadtxt('/home/guillaume/Projects/pytorch-learning/data/wine.csv',
                        delimiter=',', dtype=np.float32, skiprows=1)

        # Splitting the dataset in separate X, y arrays without converting to tensor
        self.X = xy[:, 1:]
        self.y = xy[:, [0]].astype(np.int64) -1 # using 0 in array to have shape (178, 1)
        
        # Dataset attributes
        self.n_samples, self.n_features = self.X.shape
        self.transform = transform

    def __getitem__(self, index):
        # Function allowing user to index in dataset, returns a tuple -> (features, label)
        sample = self.X[index], self.y[index]
        
        # Optional transform
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def __len__(self):
        # Function allowing user to use len(dataset) to have n_samples
        return self.n_samples

# %%
# Creating custom transform
class ToTensor:
    
    # Must always define __call__
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

# %%
# Testing ToTensor transformation
wineTensor= WineDataset(transform=ToTensor())
first_entry = wineTensor[0]
wt_features, wt_labels = first_entry
print(type(wt_features), type(wt_labels))
# %%
# Implementing scaling transform
class ScalingTransform:
    
    # Using init since we need a local variable
    def __init__(self, factor):
        self.factor = factor
        
    def __call__(self, sample):
        
        inputs, targets = sample
        
        # The transformation only scales the features
        inputs *= self.factor
        
        return inputs, targets
        
# %%
# We can use torchvision.transform.Compose to use a list of multiple transformation

composed = torchvision.transforms.Compose([ToTensor(), ScalingTransform(factor=3)])

# Testing out the composed transformation
wineScaled = WineDataset(transform=composed)
first_entry = wineScaled[0]
ws_features, ws_labels = first_entry
print(f'WineTensor features : {wt_features}\nWineTensor Targets : {wt_labels}\n')
print(f'WineScaled features : {ws_features}\nWineScaled Targets : {ws_labels}\n')

# %%
