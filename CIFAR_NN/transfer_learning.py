# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

# %%
# Setting up hyper paramters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lr = 0.001
num_epochs = 3
batch_size = 25
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#%%
# Loading the data in Dataloaders
train_set = datasets.CIFAR10(root='./data', train=True, 
                             transform=transforms.ToTensor(), download=True)

test_set = datasets.CIFAR10(root='./data', train=False,
                            transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                          shuffle=True, num_workers=2)

test_loader = DataLoader(dataset=test_set, batch_size=batch_size,
                         num_workers=2)

# %%
model = models.resnet50(pretrained=True, progress=True)

# Modyfying the last fully connected layer
n_features = model.fc.in_features
model.fc = nn.Linear(n_features, out_features=10)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr)

# %%
# Training loop
for epoch in range(num_epochs):
    
    for i, (images, labels) in enumerate(train_loader):
        
        images = images.to(device)
        labels = labels.to(device)
        
        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if (i + 1) % 100 == 0:
            print(
                f'epoch {epoch + 1}, step {i + 1}/2000, loss {loss.item():.4f}'
            )

# %%
# Validation test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
