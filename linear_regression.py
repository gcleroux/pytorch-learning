# Creating a Linear regression model entirely from PyTorch modules

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
import matplotlib.pyplot as plt

#%%
# Preparing the data

# Loading the data in numpy arrays
X_numpy, y_numpy = datasets.make_regression(n_samples=200, n_features=1, noise=42, random_state=42)

# Converting numpy arrays to tensors
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

# Reshaping the tensor to have a vector column 
y = y.reshape([-1, 1])

n_samples, n_features = X.shape

#%%
# Creating the model class
class LinearRegression(nn.Module):
    
    def __init__(self, lr=0.01):
        super(LinearRegression, self).__init__()
        
        # Defining the layer
        self.linear = nn.Linear(in_features=n_features, out_features=1)
        self.lr = lr
        
    def forward(self, samples):
        # Forward pass is just the linear regression
        return self.linear(samples)

# %%

# Creating the model
model = LinearRegression(lr=0.1)

# Choosing the loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), model.lr)
# %%

# Implementing the training loop

for epoch in range(40):
    
    # Forward pass
    y_predicted = model(X)
    
    # Calculating the loss
    loss = criterion(y, y_predicted)
    
    # Calculating the gradients through backpropagation
    loss.backward()
    
    # Updating the weights of the model
    optimizer.step()
    
    # Reseting the gradients for next epoch
    optimizer.zero_grad()
    
    # Printing model loss at every 10 epoch
    if epoch % 5 == 0:
        print(f'epoch {epoch + 1}: loss = {loss.item():.4f}')

# %%
# Plotting the results to visualize data
predictions = model(X).detach().numpy()
plt.scatter(X_numpy, y_numpy, s=3, c='red')
plt.plot(X_numpy, predictions, 'b')
plt.show()

# %%
