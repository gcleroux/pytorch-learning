#%%
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import _message_with_time
import torch
import torch.nn as nn
from IrisModel import IrisModel

#%%
# Loading the dataset and creating training/testing splits
ds = datasets.load_iris()

X_np, y_np = ds.data, ds.target

n_samples, n_features, n_classes = X_np.shape[0], X_np.shape[1], len(np.unique(y_np))

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X_np, y_np, test_size=0.2, random_state=42, shuffle=True)

# Loading the arrays in tensors
X_train = torch.from_numpy(X_train_np.astype(np.float32))
X_test = torch.from_numpy(X_test_np.astype(np.float32))
y_train = torch.from_numpy(y_train_np.astype(np.int64))
y_test = torch.from_numpy(y_test_np.astype(np.int64))


# %%
# Creating the model
model = IrisModel(n_features=n_features, n_classes=n_classes, lr=0.01)

# Creating the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), model.lr)

# %%
# Training loop

for epoch in range(2000):
    
    # Forward pass
    y_predicted = model(X_train)
    
    # Calculating the loss
    loss = criterion(y_predicted, y_train)
    
    # Calculating the gradients
    loss.backward()
    
    # Updating the weights
    optimizer.step()
    
    # Emptying the gradients
    optimizer.zero_grad()
    
    if epoch % 50 == 0:
        # Formatting the predictions tensor
        predictions = torch.tensor([torch.argmax(y) for y in y_predicted])
        
        print(f'Epoch {epoch + 1}\nLoss = {loss.item():.4f}\nAccuracy = {model.accuracy(y_train, predictions):.4f}\n')
    
    
# %%
# Validation test
with torch.no_grad():
    y_predicted = model(X_test)
    predictions = torch.tensor([torch.argmax(y) for y in y_predicted])

    print('***************\nModel Results\n***************')
    print(f'Accuracy = {model.accuracy(y_test, predictions):.4f}')
