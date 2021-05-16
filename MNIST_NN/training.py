# %%
import torch
from torch import optim
from torch._C import device
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import NeuralNet

# %%
# Hyper parameters
input_size = 28 * 28    # Size of a picture
hidden_size = 142       # Why not?
num_classes = 10        # 0 - 9
num_epochs = 3
batch_size = 100
lr = 0.001

# %%
# Importing MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                           transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                          transform=transforms.ToTensor())

# Creating the dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                          shuffle=True, num_workers=2)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                         num_workers=2)

# %%
# Looking at one element of the dataset 
example = iter(train_loader)
samples, labels = example.next()
print(samples.shape, labels.shape)

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
plt.show()

# %%
# Traning the model
model = NeuralNet(input_size=input_size, hidden_size=hidden_size, output_size=num_classes, lr=lr)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

for epoch in range(num_epochs):
    
    for i, (samples, labels) in enumerate(train_loader):
        
        # Reshaping the images
        samples = samples.reshape(-1, input_size)
    
        predictions = model(samples)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if i % 100 == 0:
            print(
                f'epoch {epoch + 1}, step {i + 1}/600, loss = {loss.item():.4f}'
            )
        
# %%
# Validation test
with torch.no_grad():
    
    for images, labels in test_loader:
        
        images = images.reshape(-1, input_size)
        predictions = model(images)
        
        # Calculating the accuracy
        _, predictions = torch.max(predictions, 1)
        accuracy = ((predictions == labels).sum() / 100).item()
        
    print('***************\nValidation test\n***************')
    print(f'Model accuracy = {accuracy:.4f}')
