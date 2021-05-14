#%%
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from BCModel import BCModel

# %%
# Creating the dataset
dset = datasets.load_breast_cancer()

X_numpy, y_numpy = dset.data, dset.target

n_samples, n_features = X_numpy.shape

# Making a training and test split
X_train_numpy, X_test_numpy, y_train_numpy, y_test_numpy = train_test_split(
    X_numpy, y_numpy, test_size=0.2, random_state=42, shuffle=True)

X_train = torch.from_numpy(X_train_numpy.astype(np.float32))
X_test = torch.from_numpy(X_test_numpy.astype(np.float32))

y_train = torch.from_numpy(y_train_numpy.astype(np.float32))
y_train = y_train.reshape([-1, 1])

y_test = torch.from_numpy(y_test_numpy.astype(np.float32))
y_test = y_test.reshape([-1, 1])


# %%
# Creating the model
model = BCModel(n_features=n_features)

# Creating the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), model.lr)

# %%
# Training loop

for epoch in range(5000):

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

    if epoch % 500 == 0:

        print(
            f'epoch {epoch + 1}: loss = {loss.item():.4f}\naccuracy = {model.accuracy(y_predicted, y_train):.4f}\n')

#%%
# Validation test
with torch.no_grad():

    y_predicted = model(X_test)

    print('***************\nModel Results\n***************')
    print(f'Accuracy = {model.accuracy(y_predicted, y_test):.4f}')
