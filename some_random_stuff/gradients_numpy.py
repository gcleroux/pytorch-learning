#%%
# Implementation of the linear regression completely from scratch
import numpy as np
from numpy.core.fromnumeric import shape
from torch.autograd import backward

# Implementation of a linear regression 
# f = w * x -> 2 * x

X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)

w = 0.0

def forward(x):
    return w * x

# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()


# Gradient of the loss 
# Loss MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x * (w*x - y)
def gradient(x, y, y_predicted):
    return np.dot(2 * x, y_predicted - y).mean()

print(f'Predictions before training: f(5) = {forward(5):.3f}')

# Training 
lr = 0.01
n_iters = 20

for epoch in range(n_iters):
    
    # Prediction
    y_pred = forward(X)
    
    # Calculating the loss
    l = loss(Y, y_pred)
    
    # Calculating the gradients
    gradients = gradient(X, Y, y_pred)
    
    # update the weights
    w -= lr * gradients
    
    if epoch % 1 == 0:
        print(f'epoch : {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')
        
print(f'Predictions after training: f(5) = {forward(5):.3f}')

# %%
import torch

# Implementation with the autograd to calculate the gradients automatically

# Implementation of a linear regression
# f = w * x -> 2 * x

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# Now we need to specify that we need to calculate the gradients of the weights 
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return w * x

# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()


print(f'Predictions before training: f(5) = {forward(5):.3f}')

# Training
lr = 0.01
n_iters = 100

for epoch in range(n_iters):

    # Prediction
    y_pred = forward(X)

    # Calculating the loss
    l = loss(Y, y_pred)

    # Calculating the gradients with backward pass
    l.backward() # dl/dw

    # update the weights, but we need to make sure the update is not calculated in the gradient calculations 
    with torch.no_grad():
        w -= lr * w.grad
        
    # empty the gradients
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch : {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Predictions after training: f(5) = {forward(5):.3f}')

# %%
import torch
import torch.nn as nn

# Automatic implementation of the loss fonction and the update of the weights in the pipeline

# Implementation of a linear regression
# f = w * x -> 2 * x

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# Now we need to specify that we need to calculate the gradients of the weights
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


def forward(x):
    return w * x

# Training
lr = 0.01
n_iters = 100

# We give the loss function to the model
loss = nn.MSELoss()

# We create an optimizer in charge of updating the weights
optimizer = torch.optim.SGD([w], lr=lr)

print(f'Predictions before training: f(5) = {forward(5):.3f}')

for epoch in range(n_iters):

    # Prediction
    y_pred = forward(X)

    # Calculating the loss
    l = loss(Y, y_pred)

    # Calculating the gradients with backward pass
    l.backward()  # dl/dw

    # Automatic weights update
    optimizer.step()

    # emptying the gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'epoch : {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Predictions after training: f(5) = {forward(5):.3f}')

# %%

# Automatic implementation of the forward pass

# Implementation of a linear regression
# f = w * x -> 2 * x

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape

# Training paramteters
lr = 0.1
n_iters = 300

# Defining the forward pass, both input/output = 1
model = nn.Linear(in_features=n_features, out_features=n_features)

# We give the loss function to the model
loss = nn.MSELoss()

# We create an optimizer in charge of updating the weights
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

print(f'Predictions before training: f(5) = {model(X_test).item():.3f}')

for epoch in range(n_iters):

    # Prediction
    y_pred = model(X)

    # Calculating the loss
    l = loss(Y, y_pred)

    # Calculating the gradients with backward pass
    l.backward()  # dl/dw

    # Automatic weights update
    optimizer.step()

    # emptying the gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        # Unpacking the parameters tensors 
        [w, b] = model.parameters()
        print(f'epoch : {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Predictions after training: f(5) = {model(X_test).item():.3f}')


# %%
