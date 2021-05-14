# Creating a Logistic regression model entirely from PyTorch modules

# To create a model, we have to implements these steps in order:
#
#   1- Design the model (input size, output size, forward pass)
#   2- Construct the loss and optimizer
#   3- Construct the training loop

#%%
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# %%
# Creating the dataset
dset = datasets.load_iris()

X_numpy, y_numpy = dset.data, dset.target

n_samples, n_features = X_numpy.shape

# Making a training and test split
X_train_numpy, X_test_numpy, y_train_numpy, y_test_numpy = train_test_split(
    X_numpy, y_numpy, test_size=0.2, random_state=42, shuffle=True)

# I found that, no clue what its supposed to do
"""
sc = StandardScaler()

X_train_numpy = sc.fit_transform(X_train_numpy)
X_test_numpy = sc.fit_transform(X_test_numpy)
"""

X_train = torch.from_numpy(X_train_numpy.astype(np.float32))
X_test = torch.from_numpy(X_test_numpy.astype(np.float32))

y_train = torch.from_numpy(y_train_numpy.astype(np.int64))

y_test = torch.from_numpy(y_test_numpy.astype(np.int64))

# %%
# Creating the model


class LogisticRegression(nn.Module):

    def __init__(self, lr=0.0001, n_classes=3):
        super(LogisticRegression, self).__init__()
        self.lr = lr
        self.n_classes = n_classes

        # Defining the layers
        self.linear = nn.Linear(in_features=n_features, out_features=self.n_classes)

    def forward(self, X):

        return self.linear(X)
        '''
        y_predicted = torch.sigmoid(self.linear(X))
        return y_predicted.flatten()
        '''


# %%
model = LogisticRegression()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), model.lr)

# %%

# Training loop

for epoch in range(2000):

    # Forward pass
    y_predicted = model(X_train)

    # Calulating the loss
    loss = criterion(y_predicted, y_train)

    # Calculating the gradients
    loss.backward()

    # Updating the gradients
    optimizer.step()

    # Clearing out the gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'epoch {epoch + 1}: loss = {loss.item():.4f}')

# %%
# Testing the model
